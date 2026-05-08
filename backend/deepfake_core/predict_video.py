"""
视频预测完整流程
从视频中提取人脸 -> 预测每个人脸 -> 融合为视频级结果
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json
import tempfile
import shutil
import os
import joblib

import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import mediapipe as mp

from . import config
from .model import load_model_for_inference
from .utils import setup_logger, get_device, load_json, save_json
from .fusion import fusion_analysis, compare_fusion_methods
from .fft import analyze_frequency_from_frames
from .sync import analyze_av_sync
from .llm import generate_llm_explanation
from .gradcam import generate_gradcam


_MEDIAPIPE_MODEL_CACHE = None


def _get_mediapipe_model_asset(model_path: Path) -> Path:
    """Return an ASCII-only model path for MediaPipe Tasks when needed."""
    global _MEDIAPIPE_MODEL_CACHE

    if str(model_path).isascii():
        return model_path

    if _MEDIAPIPE_MODEL_CACHE is not None and Path(_MEDIAPIPE_MODEL_CACHE).exists():
        return Path(_MEDIAPIPE_MODEL_CACHE)

    cache_dir = Path(tempfile.gettempdir()) / "deepfake_mediapipe_model"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / model_path.name

    if not cached_path.exists():
        shutil.copy2(model_path, cached_path)

    _MEDIAPIPE_MODEL_CACHE = cached_path
    return cached_path


def extract_frames_from_video(
    video_path: Path,
    frame_interval: int = None,
    max_frames: int = None
) -> List[np.ndarray]:
    """
    从视频中提取帧

    Args:
        video_path: 视频路径
        frame_interval: 帧间隔
        max_frames: 最大帧数

    Returns:
        帧列表 (numpy arrays)
    """
    if frame_interval is None:
        frame_interval = config.FRAME_INTERVAL

    if max_frames is None:
        max_frames = config.MAX_FRAMES_PER_VIDEO

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_count = 0
    extracted_count = 0

    while cap.isOpened() and extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # 转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            extracted_count += 1

        frame_count += 1

    cap.release()
    logging.info(f"从视频提取了 {len(frames)} 帧")
    return frames


def detect_and_crop_faces(
    frames: List[np.ndarray],
    model_path: Path = None,
    margin: float = None
) -> List[Tuple[Image.Image, int]]:
    """
    检测并裁剪人脸

    Args:
        frames: 帧列表
        model_path: MediaPipe 模型路径
        margin: 边距比例

    Returns:
        人脸图像列表 (PIL Images)
    """
    if model_path is None:
        model_path = config.MEDIAPIPE_MODEL_PATH

    if margin is None:
        margin = config.FACE_MARGIN

    faces = []  # list of (PIL.Image, frame_index)

    def _detect_with_haar(frame_list: List[np.ndarray]) -> List[Tuple[Image.Image, int]]:
        detected_faces = []
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        for idx, frame in enumerate(frame_list):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            detections = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30)
            )
            if len(detections) == 0:
                continue

            x, y, w, h = max(detections, key=lambda d: d[2] * d[3])
            x_min = max(0, int(x - w * margin))
            y_min = max(0, int(y - h * margin))
            x_max = min(frame.shape[1], int(x + w * (1 + margin)))
            y_max = min(frame.shape[0], int(y + h * (1 + margin)))

            face_crop = frame[y_min:y_max, x_min:x_max]
            if face_crop.size == 0:
                continue
            detected_faces.append((Image.fromarray(face_crop), idx))

        return detected_faces

    # 优先使用 MediaPipe Tasks；如果 Windows 路径编码或运行时初始化失败，再回退 Haar。
    try:
        mp_model_path = _get_mediapipe_model_asset(model_path)
        base_options = mp.tasks.BaseOptions(model_asset_path=str(mp_model_path))
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5
        )
        detector = mp.tasks.vision.FaceDetector.create_from_options(options)

        for idx, frame in enumerate(frames):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = detector.detect(mp_image)

            if not detection_result.detections:
                continue

            largest_detection = max(
                detection_result.detections,
                key=lambda d: d.bounding_box.width * d.bounding_box.height
            )

            bbox = largest_detection.bounding_box
            h, w = frame.shape[:2]

            x_min = max(0, int(bbox.origin_x - bbox.width * margin))
            y_min = max(0, int(bbox.origin_y - bbox.height * margin))
            x_max = min(w, int(bbox.origin_x + bbox.width * (1 + margin)))
            y_max = min(h, int(bbox.origin_y + bbox.height * (1 + margin)))

            face_crop = frame[y_min:y_max, x_min:x_max]
            if face_crop.size == 0:
                continue
            faces.append((Image.fromarray(face_crop), idx))

        detector.close()

        # 如果 MediaPipe 初始化成功但一个人脸都没找到，再用 Haar 兜底一次。
        if not faces:
            logging.warning("MediaPipe 未检测到人脸，改用 OpenCV Haar cascade 兜底")
            faces = _detect_with_haar(frames)
    except Exception as e:
        logging.warning(f"MediaPipe Tasks initialization failed, falling back to OpenCV Haar cascade: {e}")
        faces = _detect_with_haar(frames)

    logging.info(f"检测到 {len(faces)} 张人脸")
    return faces


def predict_faces(
    faces: List[Image.Image],
    model_path: Path = None,
    device: torch.device = None
) -> List[Dict[str, float]]:
    """
    预测人脸

    Args:
        faces: 人脸图像列表
        model_path: 模型路径
        device: 计算设备

    Returns:
        预测结果列表
    """
    if model_path is None:
        model_path = config.BEST_MODEL_PATH

    if device is None:
        device = get_device()

    # 加载模型配置
    model_config_path = config.MODEL_CONFIG_PATH
    if model_config_path.exists():
        model_config = load_json(model_config_path)
        model_name = model_config.get('model_name', config.MODEL_NAME)
        image_size = model_config.get('image_size', config.IMAGE_SIZE)
    else:
        model_name = config.MODEL_NAME
        image_size = config.IMAGE_SIZE

    # 加载模型
    logging.info(f"加载模型: {model_path}")
    model, _ = load_model_for_inference(str(model_path), model_name=model_name, device=device)

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])

    predictions = []

    with torch.no_grad():
        for face in faces:
            # 预处理
            input_tensor = transform(face).unsqueeze(0).to(device)

            # 预测
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]

            # 确保字段含义一致：'fake_prob' 表示 P(fake), 'real_prob' 表示 P(real)
            fake_prob = probs[0].item()
            real_prob = probs[1].item()
            predicted_label = config.CLASS_NAMES[probs.argmax().item()]

            predictions.append({
                'real_prob': real_prob,
                'fake_prob': fake_prob,
                'predicted_label': predicted_label
            })

    logging.info(f"完成 {len(predictions)} 张人脸的预测")
    return predictions


def predict_video(
    video_path: Path,
    model_path: Path = None,
    fusion_method: str = "weighted_average",
    output_path: Path = None,
    device: torch.device = None,
    enable_fft: bool = False,
    enable_sync: bool = False,
    enable_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
) -> Dict:
    """
    完整的视频预测流程

    Args:
        video_path: 视频路径
        model_path: 模型路径
        fusion_method: 融合方法
        output_path: 输出 JSON 路径
        device: 计算设备

    Returns:
        预测结果字典
    """
    logging.info(f"开始预测视频: {video_path}")

    # 检查文件是否存在
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 1. 提取帧
    logging.info("步骤 1/3: 提取视频帧...")
    frames = extract_frames_from_video(video_path)

    if not frames:
        logging.warning("未能从视频中提取帧")
        return {
            'video_path': str(video_path),
            'error': 'no_frames_extracted',
            'video_label': 'unknown',
            'video_fake_prob': 0.5
        }

    # 2. 检测和裁剪人脸
    logging.info("步骤 2/3: 检测和裁剪人脸...")
    faces_with_idx = detect_and_crop_faces(frames)
    # faces_with_idx: List[(PIL.Image, frame_index)]
    faces = [f for (f, _) in faces_with_idx]
    face_frame_indices = [idx for (_, idx) in faces_with_idx]

    if not faces:
        logging.warning("未能从视频中检测到人脸")
        return {
            'video_path': str(video_path),
            'error': 'no_faces_detected',
            'video_label': 'unknown',
            'video_fake_prob': 0.5,
            'num_frames': len(frames)
        }

    # 3. 预测人脸
    logging.info("步骤 3/3: 预测人脸...")
    face_predictions = predict_faces(faces, model_path, device)

    # 4. 可选 Module B 分析
    frequency_score = None
    sync_score = None
    llm_analysis = None

    module_b_details = {}

    try:
        freq_result = analyze_frequency_from_frames(frames)
        module_b_details['fft'] = freq_result
        if freq_result.get('available'):
            frequency_score = freq_result.get('frequency_score')
    except Exception as e:
        logging.warning(f"FFT 模块执行失败: {e}")
        module_b_details['fft'] = {'available': False, 'reason': str(e)}

    try:
        sync_result = analyze_av_sync(video_path)
        module_b_details['sync'] = sync_result
        if sync_result.get('available'):
            sync_score = sync_result.get('sync_score')
    except Exception as e:
        logging.warning(f"Sync 模块执行失败: {e}")
        module_b_details['sync'] = {'available': False, 'reason': str(e)}

    # Compute suspicious face/frame indices (top-k faces by fake_prob)
    suspicious_frame_indices = []
    gradcam_paths = {}
    try:
        if face_predictions:
            fake_probs = [p['fake_prob'] for p in face_predictions]
            # choose top-k (k = max(1, 10% of faces) but at most 5)
            k = max(1, int(len(fake_probs) * 0.10))
            k = min(k, 5)
            topk_idx = list(np.argsort(fake_probs)[-k:][::-1])
            for face_idx in topk_idx:
                if face_idx < len(face_frame_indices):
                    suspicious_frame_indices.append(face_frame_indices[face_idx])

            # generate Grad-CAM overlays for suspicious faces (save temporary face images)
            import tempfile
            from pathlib import Path
            tmp_dir = Path(tempfile.gettempdir()) / "deepfake_gradcam_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            for i, face_idx in enumerate(topk_idx):
                if face_idx < len(faces):
                    face_img = faces[face_idx]
                    tmp_path = tmp_dir / f"sus_face_{Path(str(video_path)).stem}_{i}.jpg"
                    face_img.save(tmp_path)
                    try:
                        out = generate_gradcam(tmp_path, model_path=model_path, output_dir=tmp_dir, device=device)
                        gradcam_paths[face_frame_indices[face_idx]] = {k: str(v) for k, v in out.items()}
                    except Exception as e:
                        logging.warning(f"生成 Grad-CAM 失败: {e}")
    except Exception as e:
        logging.warning(f"生成可疑帧/GradCAM 过程异常: {e}")

    # 5. 融合结果
    logging.info("融合结果...")
    result = fusion_analysis(
        face_predictions,
        fusion_method,
        frequency_score=frequency_score,
        sync_score=sync_score,
        llm_analysis=None,
    )

    # 尝试加载后融合校准器（如果已训练并保存）并生成校准概率
    try:
        calib_path = config.MODELS_DIR / "fusion_calibrator.pkl"
        if calib_path.exists():
            try:
                calib = joblib.load(calib_path)

                feat_video_prob = float(result.get('video_fake_prob', 0.5))
                feat_freq = frequency_score if frequency_score is not None else float('nan')
                feat_sync = sync_score if sync_score is not None else float('nan')
                feat_freq_missing = 1 if frequency_score is None else 0
                feat_sync_missing = 1 if sync_score is None else 0

                X = pd.DataFrame([{
                    'video_fake_prob': feat_video_prob,
                    'frequency_score': feat_freq,
                    'sync_score': feat_sync,
                    'frequency_missing': feat_freq_missing,
                    'sync_missing': feat_sync_missing,
                }])

                proba = calib.predict_proba(X)[0]
                # 训练时我们以 "fake" 为正类 (y=1 表示 fake)，因此 proba[1] 为 P(fake)
                calibrated_fake_prob = float(proba[1])
                calibrated_label = 'fake' if calibrated_fake_prob >= config.THRESHOLD else 'real'

                result['calibrated'] = {
                    'calibrated_fake_prob': calibrated_fake_prob,
                    'calibrated_label': calibrated_label
                }
            except Exception as e:
                logging.warning(f"加载/应用融合校准器失败: {e}")
    except Exception:
        # 任何异常都不要影响主预测流程
        pass

    if module_b_details:
        result['module_b_details'] = module_b_details

    if enable_llm:
        try:
            # pass richer context to LLM
            llm_analysis = generate_llm_explanation(
                result,
                model=llm_model,
                suspicious_frames=suspicious_frame_indices or None,
                gradcam_paths=gradcam_paths or None,
                frequency_summary=module_b_details.get('fft'),
                sync_summary=module_b_details.get('sync'),
            )
            result['module_b']['llm_analysis'] = llm_analysis
        except Exception as e:
            logging.warning(f"LLM 模块执行失败: {e}")
            result['module_b']['llm_analysis'] = {
                'provider': 'none',
                'model': 'none',
                'text': f'LLM 模块异常: {e}'
            }

    # 添加视频信息
    result['video_path'] = str(video_path)
    result['num_frames'] = len(frames)

    # 保存结果
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(result, output_path)
        logging.info(f"结果已保存: {output_path}")

    logging.info(f"视频预测完成: {result['video_label']} (fake_prob={result['video_fake_prob']:.3f})")
    return result


def batch_predict_videos(
    video_dir: Path,
    model_path: Path = None,
    fusion_method: str = "weighted_average",
    output_dir: Path = None,
    device: torch.device = None,
    enable_fft: bool = False,
    enable_sync: bool = False,
    enable_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
) -> List[Dict]:
    """
    批量预测视频

    Args:
        video_dir: 视频目录
        model_path: 模型路径
        fusion_method: 融合方法
        output_dir: 输出目录
        device: 计算设备

    Returns:
        预测结果列表
    """
    # 查找所有视频文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f'*{ext}'))

    logging.info(f"找到 {len(video_files)} 个视频文件")

    results = []

    for video_path in video_files:
        try:
            # 输出路径
            if output_dir:
                output_path = output_dir / f"{video_path.stem}_result.json"
            else:
                output_path = None

            # 预测
            result = predict_video(
                video_path,
                model_path,
                fusion_method,
                output_path,
                device,
                enable_fft,
                enable_sync,
                enable_llm,
                llm_model,
            )
            results.append(result)

        except Exception as e:
            logging.error(f"预测视频 {video_path} 失败: {e}")
            results.append({
                'video_path': str(video_path),
                'error': str(e),
                'video_label': 'error'
            })

    # 保存汇总结果
    if output_dir:
        summary_path = output_dir / "batch_summary.json"
        save_json(results, summary_path)
        logging.info(f"批量预测汇总已保存: {summary_path}")

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="视频 Deepfake 检测")
    parser.add_argument("--video", type=str, help="视频路径")
    parser.add_argument("--video_dir", type=str, help="视频目录（批量预测）")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--fusion_method", type=str, default="weighted_average",
                        choices=["average", "max", "top_vote", "weighted_average"],
                        help="融合方法")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（批量预测）")
    parser.add_argument("--enable_fft", action="store_true", help="启用 FFT/DCT 频域分析")
    parser.add_argument("--enable_sync", action="store_true", help="启用音视频同步检测")
    parser.add_argument("--enable_llm", action="store_true", help="启用 LLM 自动解释")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="LLM 模型名称")

    args = parser.parse_args()

    # 设置日志
    setup_logger()

    # 检查参数
    if not args.video and not args.video_dir:
        parser.error("必须指定 --video 或 --video_dir")

    # 转换路径
    model_path = Path(args.model_path) if args.model_path else None
    device = get_device()

    if args.video:
        # 单个视频预测
        video_path = Path(args.video)
        output_path = Path(args.output) if args.output else None

        result = predict_video(
            video_path,
            model_path,
            args.fusion_method,
            output_path,
            device,
            args.enable_fft,
            args.enable_sync,
            args.enable_llm,
            args.llm_model,
        )

        # 打印结果
        print(json.dumps(result, indent=2, ensure_ascii=False))

    else:
        # 批量预测
        video_dir = Path(args.video_dir)
        output_dir = Path(args.output_dir) if args.output_dir else None

        results = batch_predict_videos(
            video_dir,
            model_path,
            args.fusion_method,
            output_dir,
            device,
            args.enable_fft,
            args.enable_sync,
            args.enable_llm,
            args.llm_model,
        )

        # 打印汇总
        print(f"\n批量预测完成: {len(results)} 个视频")
        fake_count = sum(1 for r in results if r.get('video_label') == 'fake')
        real_count = sum(1 for r in results if r.get('video_label') == 'real')
        print(f"  - FAKE: {fake_count}")
        print(f"  - REAL: {real_count}")


if __name__ == "__main__":
    main()
