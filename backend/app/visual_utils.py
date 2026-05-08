from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def generate_fft_spectrum_image(
    video_path: str,
    output_dir: Path,
    frame_interval: int = 10,
    max_frames: int = 30,
) -> str | None:
    """
    Generate an average FFT spectrum image from sampled video frames.

    Returns the saved image path as string.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fft_spectrum.png"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    spectra = []
    frame_count = 0
    saved_count = 0

    while saved_count < max_frames:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (256, 256))

            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.log1p(np.abs(fft_shift))

            spectra.append(magnitude)
            saved_count += 1

        frame_count += 1

    cap.release()

    if not spectra:
        return None

    avg_spectrum = np.mean(np.stack(spectra), axis=0)

    avg_spectrum = avg_spectrum - avg_spectrum.min()
    if avg_spectrum.max() > 0:
        avg_spectrum = avg_spectrum / avg_spectrum.max()

    img = (avg_spectrum * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(output_path)

    return str(output_path)
