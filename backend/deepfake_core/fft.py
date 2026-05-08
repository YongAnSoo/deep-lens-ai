"""
FFT/DCT frequency domain analysis.
Provides frame-level spectral anomaly features and a video-level risk score.
"""

from typing import Dict, List

import cv2
import numpy as np


def _normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max - x_min < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def _high_freq_ratio_fft(gray_frame: np.ndarray, radius_ratio: float = 0.35) -> float:
    h, w = gray_frame.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_r = np.sqrt(cy * cy + cx * cx)
    threshold = radius_ratio * max_r

    fft2 = np.fft.fft2(gray_frame)
    fft_shift = np.fft.fftshift(fft2)
    mag = np.abs(fft_shift)

    high_mask = dist >= threshold
    high_energy = float(np.sum(mag[high_mask]))
    total_energy = float(np.sum(mag) + 1e-8)
    return high_energy / total_energy


def _high_freq_ratio_dct(gray_frame: np.ndarray, keep_low: float = 0.25) -> float:
    x = gray_frame.astype(np.float32)
    x = _normalize_01(x)
    dct_map = cv2.dct(x)
    power = np.abs(dct_map)

    h, w = power.shape
    lh, lw = max(1, int(h * keep_low)), max(1, int(w * keep_low))

    low_energy = float(np.sum(power[:lh, :lw]))
    total_energy = float(np.sum(power) + 1e-8)
    high_energy = max(0.0, total_energy - low_energy)
    return high_energy / total_energy


def analyze_frequency_from_frames(frames: List[np.ndarray]) -> Dict:
    """
    Analyze frequency anomalies from RGB frames and return a risk score in [0, 1].
    """
    if not frames:
        return {
            "available": False,
            "frequency_score": None,
            "reason": "no_frames"
        }

    fft_ratios = []
    dct_ratios = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        fft_ratios.append(_high_freq_ratio_fft(gray))
        dct_ratios.append(_high_freq_ratio_dct(gray))

    fft_arr = np.asarray(fft_ratios, dtype=np.float32)
    dct_arr = np.asarray(dct_ratios, dtype=np.float32)

    fft_mean = float(np.mean(fft_arr))
    dct_mean = float(np.mean(dct_arr))
    fft_std = float(np.std(fft_arr))
    dct_std = float(np.std(dct_arr))

    # Empirical scaling so normal videos center around lower risk.
    raw_score = 0.5 * fft_mean + 0.5 * dct_mean
    frequency_score = float(np.clip((raw_score - 0.35) / 0.35, 0.0, 1.0))

    # Simple per-frame anomaly detection: mark frames > mean + 1*std
    anomaly_indices = []
    for i, (fval, dval) in enumerate(zip(fft_arr.tolist(), dct_arr.tolist())):
        if (fval > fft_mean + fft_std) or (dval > dct_mean + dct_std):
            anomaly_indices.append(i)

    return {
        "available": True,
        "frequency_score": frequency_score,
        "fft_high_freq_ratio_mean": fft_mean,
        "dct_high_freq_ratio_mean": dct_mean,
        "fft_high_freq_ratio_std": fft_std,
        "dct_high_freq_ratio_std": dct_std,
        "fft_ratios": fft_arr.tolist(),
        "dct_ratios": dct_arr.tolist(),
        "anomaly_frame_indices": anomaly_indices,
        "num_frames": len(frames)
    }
