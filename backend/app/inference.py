import os
import inspect
from pathlib import Path
from typing import Dict, Any


def patch_opencv_cascade_classifier() -> None:
    try:
        import cv2

        original_cascade_classifier = cv2.CascadeClassifier
        custom_haar_path = os.getenv(
            "HAAR_CASCADE_PATH",
            r"C:\deeplens_runtime\haarcascade_frontalface_default.xml",
        )

        def fixed_cascade_classifier(path=None, *args, **kwargs):
            use_path = path

            if path is not None:
                path_text = str(path).replace("\\", "/").lower()
                if "haarcascade_frontalface_default.xml" in path_text:
                    if Path(custom_haar_path).exists():
                        use_path = custom_haar_path

            return original_cascade_classifier(str(use_path), *args, **kwargs)

        cv2.CascadeClassifier = fixed_cascade_classifier

    except Exception:
        pass


patch_opencv_cascade_classifier()

from deepfake_core import predict_video as predict_video_module


def _get_predict_function():
    if hasattr(predict_video_module, "predict_video"):
        return predict_video_module.predict_video

    if hasattr(predict_video_module, "predict_video_file"):
        return predict_video_module.predict_video_file

    raise RuntimeError("deepfake_core.predict_video 里找不到 predict_video 或 predict_video_file")


def _call_predict_function(video_path: str) -> Dict[str, Any]:
    predict_func = _get_predict_function()
    supported_params = inspect.signature(predict_func).parameters

    # 重点：这里故意不传 enable_llm
    # 因为你的 predict_video 内部会把 enable_llm 传给不支持的函数，导致 500
    possible_kwargs = {
        "video_path": Path(video_path),
        "video": Path(video_path),
        "model_path": None,
        "fusion_method": "weighted_average",
        "method": "weighted_average",
        "output_path": None,
        "output": None,
        "device": None,
        "enable_fft": True,
        "enable_sync": True,
    }

    kwargs = {
        key: value
        for key, value in possible_kwargs.items()
        if key in supported_params
    }

    if "video_path" not in supported_params and "video" not in supported_params:
        return predict_func(Path(video_path), **kwargs)

    return predict_func(**kwargs)


def _attach_llm_explanation(result: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from deepfake_core.llm import generate_llm_explanation

        suspicious_frames = (
            result.get("suspicious_frames")
            or result.get("suspicious_frame_indices")
            or result.get("top_suspicious_frames")
        )

        gradcam_paths = (
            result.get("gradcam_paths")
            or result.get("gradcam")
        )

        module_b = result.get("module_b") or {}

        frequency_summary = (
            result.get("frequency_summary")
            or result.get("fft_summary")
            or result.get("frequency_analysis")
            or module_b.get("frequency_summary")
            or module_b.get("frequency_analysis")
        )

        sync_summary = (
            result.get("sync_summary")
            or result.get("sync_analysis")
            or module_b.get("sync_summary")
            or module_b.get("sync_analysis")
        )

        llm_result = generate_llm_explanation(
            result,
            suspicious_frames=suspicious_frames,
            gradcam_paths=gradcam_paths,
            frequency_summary=frequency_summary,
            sync_summary=sync_summary,
        )

        result["llm_analysis"] = llm_result

    except Exception as e:
        result["llm_analysis"] = {
            "provider": "backend_fallback",
            "model": "none",
            "text": f"模型检测已完成，但 LLM 解释生成失败：{str(e)}",
        }

    return result


def predict_video_file(video_path: str) -> Dict:
    result = _call_predict_function(video_path)
    result = _attach_llm_explanation(result)
    return result
