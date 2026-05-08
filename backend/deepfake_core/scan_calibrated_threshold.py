"""
扫描 calibrated_fake_prob 阈值，并以视频级准确率为目标选择最佳阈值。

用法示例：
  python -m src.scan_calibrated_threshold \
    --input outputs/metrics/calibration_dataset.csv \
    --calibrator models/fusion_calibrator.pkl \
    --out_json outputs/metrics/calibrated_threshold_scan.json \
    --out_csv outputs/metrics/calibrated_threshold_scan.csv
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

from .fusion_calibrator import load_dataset, prepare_features


def scan_thresholds(y_true: np.ndarray, y_prob: np.ndarray, start: float, end: float, step: float) -> pd.DataFrame:
    rows = []
    t = start
    while t <= end + 1e-12:
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        rows.append({"threshold": round(float(t), 6), "accuracy": float(acc)})
        t += step

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan calibrated threshold for best video accuracy")
    parser.add_argument("--input", type=str, required=True, help="Calibration CSV path")
    parser.add_argument("--calibrator", type=str, default="models/fusion_calibrator.pkl", help="Calibrator model path")
    parser.add_argument("--start", type=float, default=0.05, help="Scan start threshold")
    parser.add_argument("--end", type=float, default=0.95, help="Scan end threshold")
    parser.add_argument("--step", type=float, default=0.01, help="Scan step")
    parser.add_argument("--out_json", type=str, default="outputs/metrics/calibrated_threshold_scan.json", help="Output summary JSON")
    parser.add_argument("--out_csv", type=str, default="outputs/metrics/calibrated_threshold_scan.csv", help="Output scan CSV")

    args = parser.parse_args()

    input_path = Path(args.input)
    calib_path = Path(args.calibrator)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibrator model not found: {calib_path}")

    df = load_dataset(input_path)
    X, y = prepare_features(df)

    model = joblib.load(calib_path)
    y_prob = model.predict_proba(X)[:, 1]

    scan_df = scan_thresholds(y.values, y_prob, args.start, args.end, args.step)

    best_row = scan_df.sort_values(["accuracy", "threshold"], ascending=[False, True]).iloc[0]
    best_threshold = float(best_row["threshold"])
    best_accuracy = float(best_row["accuracy"])

    baseline_threshold = 0.5
    baseline_pred = (y_prob >= baseline_threshold).astype(int)
    baseline_accuracy = float(accuracy_score(y.values, baseline_pred))

    summary = {
        "num_samples": int(len(y)),
        "calibrator_path": str(calib_path),
        "baseline_threshold": baseline_threshold,
        "baseline_accuracy": baseline_accuracy,
        "best_threshold": best_threshold,
        "best_accuracy": best_accuracy,
        "accuracy_gain": best_accuracy - baseline_accuracy,
    }

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_csv.write_text(scan_df.to_csv(index=False), encoding="utf-8")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
