"""
人脸检测和裁剪
使用 MediaPipe 检测人脸并裁剪保存
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp

from . import config
from . import utils


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="人脸检测和裁剪")
    parser.add_argument(
        "--face_margin",
        type=float,
        default=config.FACE_MARGIN,
        help=f"人脸周围的边距比例（默认: {config.FACE_MARGIN}）"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=config.IMAGE_SIZE,
        help=f"输出图像大小（默认: {config.IMAGE_SIZE}）"
    )
    return parser.parse_args()


class FaceDetector:
    """MediaPipe 人脸检测器封装类"""

    def __init__(self, min_detection_confidence: float = 0.5):
        """
        初始化人脸检测器

        Args:
            min_detection_confidence: 最小检测置信度
        """
        # 使用 MediaPipe 新版 API (tasks)
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        # 模型路径
        model_path = config.PROJECT_ROOT / "models" / "mediapipe" / "blaze_face_short_range.tflite"

        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe 人脸检测模型不存在: {model_path}\n"
                f"请运行以下命令下载模型:\n"
                f"curl -L -o {model_path} https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            )

        # 创建检测器选项
        base_options = python.BaseOptions(
            model_asset_path=str(model_path)
        )
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence
        )

        self.detector = vision.FaceDetector.create_from_options(options)

    def detect_faces(self, image: np.ndarray) -> Optional[List[Dict]]:
        """
        检测图像中的人脸

        Args:
            image: BGR 格式的图像

        Returns:
            人脸检测结果列表，每个结果包含 bbox 和 confidence
        """
        # 转换为 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 创建 MediaPipe Image 对象
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # 检测人脸
        detection_result = self.detector.detect(mp_image)

        if not detection_result.detections:
            return None

        faces = []
        h, w, _ = image.shape

        for detection in detection_result.detections:
            # 获取边界框
            bbox = detection.bounding_box

            # 转换为像素坐标
            x = bbox.origin_x
            y = bbox.origin_y
            width = bbox.width
            height = bbox.height

            # 获取置信度
            confidence = detection.categories[0].score if detection.categories else 0.0

            faces.append({
                'bbox': (x, y, width, height),
                'confidence': confidence
            })

        return faces

    def close(self):
        """关闭检测器"""
        self.detector.close()


def get_largest_face(faces: List[Dict]) -> Dict:
    """
    获取面积最大的人脸

    Args:
        faces: 人脸检测结果列表

    Returns:
        面积最大的人脸
    """
    largest_face = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
    return largest_face


def crop_face_with_margin(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.25
) -> np.ndarray:
    """
    裁剪人脸并添加边距

    Args:
        image: 原始图像
        bbox: 边界框 (x, y, width, height)
        margin: 边距比例

    Returns:
        裁剪后的人脸图像
    """
    h, w, _ = image.shape
    x, y, face_w, face_h = bbox

    # 计算边距
    margin_w = int(face_w * margin)
    margin_h = int(face_h * margin)

    # 扩展边界框
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(w, x + face_w + margin_w)
    y2 = min(h, y + face_h + margin_h)

    # 裁剪
    cropped = image[y1:y2, x1:x2]

    return cropped


def process_single_frame(
    frame_path: Path,
    output_dir: Path,
    label: str,
    detector: FaceDetector,
    face_margin: float = 0.25,
    image_size: int = 224
) -> Optional[Dict]:
    """
    处理单个帧：检测人脸并裁剪

    Args:
        frame_path: 帧图像路径
        output_dir: 输出目录
        label: 标签（real 或 fake）
        detector: 人脸检测器
        face_margin: 人脸边距比例
        image_size: 输出图像大小

    Returns:
        处理记录字典，如果没有检测到人脸则返回 None
    """
    # 读取图像
    image = cv2.imread(str(frame_path))

    if image is None:
        logging.warning(f"无法读取图像: {frame_path}")
        return None

    # 检测人脸
    faces = detector.detect_faces(image)

    if not faces:
        return {
            'original_frame_path': str(frame_path),
            'face_path': None,
            'label': label,
            'detected': False,
            'bbox_x': None,
            'bbox_y': None,
            'bbox_w': None,
            'bbox_h': None,
            'confidence': None
        }

    # 获取最大的人脸
    largest_face = get_largest_face(faces)
    bbox = largest_face['bbox']
    confidence = largest_face['confidence']

    # 裁剪人脸
    face_image = crop_face_with_margin(image, bbox, face_margin)

    # 调整大小
    face_image_resized = cv2.resize(face_image, (image_size, image_size))

    # 生成输出文件名（保持原文件名）
    face_filename = frame_path.name
    face_path = output_dir / face_filename

    # 保存
    cv2.imwrite(str(face_path), face_image_resized)

    return {
        'original_frame_path': str(frame_path),
        'face_path': str(face_path),
        'label': label,
        'detected': True,
        'bbox_x': bbox[0],
        'bbox_y': bbox[1],
        'bbox_w': bbox[2],
        'bbox_h': bbox[3],
        'confidence': confidence
    }


def process_frames_directory(
    frames_dir: Path,
    output_dir: Path,
    label: str,
    face_margin: float = 0.25,
    image_size: int = 224
) -> List[Dict]:
    """
    处理目录中的所有帧

    Args:
        frames_dir: 帧目录
        output_dir: 输出目录
        label: 标签（real 或 fake）
        face_margin: 人脸边距比例
        image_size: 输出图像大小

    Returns:
        处理记录列表
    """
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有帧文件
    frame_files = list(frames_dir.glob("*.jpg"))

    if not frame_files:
        logging.warning(f"目录中没有找到帧文件: {frames_dir}")
        return []

    logging.info(f"在 {frames_dir} 中找到 {len(frame_files)} 个帧")

    # 初始化人脸检测器
    detector = FaceDetector()

    records = []
    detected_count = 0

    for frame_path in tqdm(frame_files, desc=f"处理 {label} 人脸"):
        record = process_single_frame(
            frame_path=frame_path,
            output_dir=output_dir,
            label=label,
            detector=detector,
            face_margin=face_margin,
            image_size=image_size
        )

        if record:
            records.append(record)
            if record['detected']:
                detected_count += 1

    # 关闭检测器
    detector.close()

    logging.info(
        f"处理完成: {detected_count}/{len(frame_files)} 帧检测到人脸 "
        f"({detected_count/len(frame_files)*100:.1f}%)"
    )

    return records


def save_metadata(records: List[Dict], output_path: Path):
    """
    保存人脸元数据到 CSV

    Args:
        records: 元数据记录列表
        output_path: 输出 CSV 路径
    """
    if not records:
        logging.warning("没有人脸元数据可保存")
        return

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')

    logging.info(f"已保存人脸元数据: {output_path}")

    # 打印统计信息
    logging.info("\n=== 统计信息 ===")
    logging.info(f"总帧数: {len(df)}")
    logging.info(f"检测到人脸: {df['detected'].sum()}")
    logging.info(f"未检测到人脸: {(~df['detected']).sum()}")
    logging.info(f"检测率: {df['detected'].sum()/len(df)*100:.1f}%")

    if df['detected'].sum() > 0:
        detected_df = df[df['detected']]
        logging.info(f"真实人脸: {len(detected_df[detected_df['label'] == 'real'])}")
        logging.info(f"假人脸: {len(detected_df[detected_df['label'] == 'fake'])}")
        logging.info(f"平均置信度: {detected_df['confidence'].mean():.3f}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置日志
    utils.setup_logger()

    logging.info("=" * 60)
    logging.info("人脸检测和裁剪")
    logging.info("=" * 60)
    logging.info(f"人脸边距: {args.face_margin}")
    logging.info(f"输出图像大小: {args.image_size}x{args.image_size}")

    all_records = []

    # 处理真实帧
    real_frames_dir = config.FRAMES_DIR / "real"
    real_output_dir = config.FACES_DIR / "real"

    if real_frames_dir.exists():
        real_records = process_frames_directory(
            frames_dir=real_frames_dir,
            output_dir=real_output_dir,
            label="real",
            face_margin=args.face_margin,
            image_size=args.image_size
        )
        all_records.extend(real_records)
    else:
        logging.warning(f"真实帧目录不存在: {real_frames_dir}")

    # 处理假帧
    fake_frames_dir = config.FRAMES_DIR / "fake"
    fake_output_dir = config.FACES_DIR / "fake"

    if fake_frames_dir.exists():
        fake_records = process_frames_directory(
            frames_dir=fake_frames_dir,
            output_dir=fake_output_dir,
            label="fake",
            face_margin=args.face_margin,
            image_size=args.image_size
        )
        all_records.extend(fake_records)
    else:
        logging.warning(f"假帧目录不存在: {fake_frames_dir}")

    # 保存元数据
    output_path = config.METRICS_DIR / "face_metadata.csv"
    save_metadata(all_records, output_path)

    logging.info("=" * 60)
    logging.info("人脸裁剪完成！")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
