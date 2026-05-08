"""Summarize real/fake prediction folders into classification metrics."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def _load_prediction_files(pred_dir: Path, true_label: int) -> List[Dict]:
    rows: List[Dict] = []
    if not pred_dir.exists():
        return rows

    files = sorted(pred_dir.glob("*.json"))
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        score = data.get("combined_score")
        if score is None:
            score = data.get("video_fake_prob")
        if score is None:
            continue

        try:
            score = float(score)
        except Exception:
            continue

        pred_label = 1 if score >= 0.5 else 0
        rows.append(
            {
                "file": str(file_path),
                "true_label": true_label,
                "pred_label": pred_label,
                "fake_score": score,
                "video_path": data.get("video_path", ""),
            }
        )

    return rows


def summarize(real_dir: Path, fake_dir: Path, output_json: Path, output_csv: Path) -> Dict:
    rows = []
    rows.extend(_load_prediction_files(real_dir, true_label=0))
    rows.extend(_load_prediction_files(fake_dir, true_label=1))

    if not rows:
        raise RuntimeError("No valid prediction json files found in real/fake directories.")

    df = pd.DataFrame(rows)

    y_true = df["true_label"].to_numpy()
    y_pred = df["pred_label"].to_numpy()
    y_score = df["fake_score"].to_numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "num_samples": int(len(df)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        },
        "inputs": {
            "real_dir": str(real_dir),
            "fake_dir": str(fake_dir),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize prediction folders into metrics")
    parser.add_argument("--real_dir", type=str, default="outputs/predictions/real", help="Directory of real video prediction json files")
    parser.add_argument("--fake_dir", type=str, default="outputs/predictions/fake", help="Directory of fake video prediction json files")
    parser.add_argument("--output_json", type=str, default="outputs/metrics/fusion_metrics.json", help="Output metrics json path")
    parser.add_argument("--output_csv", type=str, default="outputs/metrics/fusion_per_video.csv", help="Output per-video csv path")

    args = parser.parse_args()

    metrics = summarize(
        real_dir=Path(args.real_dir),
        fake_dir=Path(args.fake_dir),
        output_json=Path(args.output_json),
        output_csv=Path(args.output_csv),
    )

    print("=" * 60)
    print("Fusion Metrics Summary")
    print("=" * 60)
    print(f"Samples:   {metrics['num_samples']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
