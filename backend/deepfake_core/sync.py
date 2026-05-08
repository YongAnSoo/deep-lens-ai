"""
Audio-visual synchronization analysis.
Extracts MFCC features from audio and compares them with mouth motion dynamics.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import mediapipe as mp


def _normalize_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mean = float(np.mean(x))
    std = float(np.std(x) + 1e-8)
    return (x - mean) / std


def _resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) == target_len:
        return x
    if len(x) == 0 or target_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    src_idx = np.linspace(0, len(x) - 1, num=len(x))
    dst_idx = np.linspace(0, len(x) - 1, num=target_len)
    return np.interp(dst_idx, src_idx, x).astype(np.float32)


def _find_ffmpeg_executable() -> Optional[str]:
    # Prefer bundled binary from imageio-ffmpeg if available.
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg and Path(ffmpeg).exists():
            return ffmpeg
    except Exception:
        pass

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "Library" / "bin" / "ffmpeg.exe"
        if candidate.exists():
            return str(candidate)

    return None


def _extract_audio_and_mfcc(video_path: Path, target_sr: int = 16000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    try:
        import librosa
    except Exception:
        return None, None, "librosa_not_available"

    audio = None
    sr = target_sr

    try:
        # librosa may require ffmpeg backend to read mp4 audio.
        audio, sr = librosa.load(str(video_path), sr=target_sr, mono=True)
    except Exception:
        ffmpeg = _find_ffmpeg_executable()
        if not ffmpeg:
            return None, None, "audio_decode_failed_no_ffmpeg"

        tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)

        try:
            cmd = [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-ac",
                "1",
                "-ar",
                str(target_sr),
                tmp_wav,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                return None, None, "audio_decode_failed_ffmpeg"

            audio, sr = librosa.load(tmp_wav, sr=target_sr, mono=True)
        except Exception:
            return None, None, "audio_decode_failed"
        finally:
            try:
                Path(tmp_wav).unlink(missing_ok=True)
            except Exception:
                pass

    if audio is None or len(audio) == 0:
        return None, None, "empty_audio"

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=512)
    # Use first MFCC coefficient dynamics as the temporal proxy.
    mfcc_signal = np.asarray(mfcc[0], dtype=np.float32)
    return np.asarray(audio, dtype=np.float32), mfcc_signal, None


def _extract_mouth_motion(video_path: Path, max_frames: int = 180, frame_stride: int = 2) -> Tuple[np.ndarray, int]:
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)

    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_mesh"):
        cap.release()
        return _extract_mouth_motion_fallback(video_path, max_frames=max_frames, frame_stride=frame_stride)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    series = []
    idx = 0

    while cap.isOpened() and len(series) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % frame_stride != 0:
            idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            idx += 1
            continue

        lm = res.multi_face_landmarks[0].landmark

        # MediaPipe face mesh landmarks:
        # 13 upper lip, 14 lower lip, 33 left eye outer, 263 right eye outer.
        up = np.array([lm[13].x, lm[13].y], dtype=np.float32)
        low = np.array([lm[14].x, lm[14].y], dtype=np.float32)
        le = np.array([lm[33].x, lm[33].y], dtype=np.float32)
        re = np.array([lm[263].x, lm[263].y], dtype=np.float32)

        mouth_open = float(np.linalg.norm(up - low))
        eye_dist = float(np.linalg.norm(le - re) + 1e-6)
        series.append(mouth_open / eye_dist)
        idx += 1

    cap.release()
    face_mesh.close()
    return np.asarray(series, dtype=np.float32), fps


def _extract_mouth_motion_fallback(video_path: Path, max_frames: int = 180, frame_stride: int = 2) -> Tuple[np.ndarray, int]:
    """
    Fallback when MediaPipe face mesh API is unavailable.
    Uses motion energy from the lower-center face region as a lip-motion proxy.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
    series = []
    prev_roi = None
    idx = 0

    while cap.isOpened() and len(series) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % frame_stride != 0:
            idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        y1, y2 = int(0.45 * h), int(0.90 * h)
        x1, x2 = int(0.25 * w), int(0.75 * w)
        roi = gray[y1:y2, x1:x2]

        if roi.size == 0:
            idx += 1
            continue

        roi = cv2.GaussianBlur(roi, (5, 5), 0)
        if prev_roi is not None and prev_roi.shape == roi.shape:
            motion = float(np.mean(cv2.absdiff(roi, prev_roi)) / 255.0)
            series.append(motion)

        prev_roi = roi
        idx += 1

    cap.release()
    return np.asarray(series, dtype=np.float32), fps


def analyze_av_sync(video_path: Path) -> Dict:
    """
    Returns sync risk score in [0, 1], where higher means higher desync risk.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return {"available": False, "sync_score": None, "reason": "video_not_found"}

    _, mfcc_signal, audio_err = _extract_audio_and_mfcc(video_path)
    if mfcc_signal is None:
        return {"available": False, "sync_score": None, "reason": audio_err}

    mouth_signal, fps = _extract_mouth_motion(video_path)
    if mouth_signal.size < 8:
        return {"available": False, "sync_score": None, "reason": "insufficient_mouth_landmarks"}

    target_len = min(len(mfcc_signal), len(mouth_signal))
    if target_len < 8:
        return {"available": False, "sync_score": None, "reason": "insufficient_temporal_samples"}

    a = _normalize_1d(_resample_1d(mfcc_signal, target_len))
    v = _normalize_1d(_resample_1d(mouth_signal, target_len))

    corr = float(np.corrcoef(a, v)[0, 1]) if target_len > 1 else 0.0
    if np.isnan(corr):
        corr = 0.0

    corr_01 = (corr + 1.0) / 2.0
    sync_score = float(np.clip(1.0 - corr_01, 0.0, 1.0))

    return {
        "available": True,
        "sync_score": sync_score,
        "temporal_correlation": corr,
        "num_samples": int(target_len),
        "fps": int(fps),
    }
