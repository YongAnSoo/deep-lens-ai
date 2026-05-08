"""
批量重算视频预测并比较三种融合方法（average, top_vote, weighted_average）。

用法示例:
  python -m src.batch_recompute_compare \
    --video_dir data/kaggle_dfdc/00 \
    --metadata data/kaggle_dfdc/00/metadata.json \
    --out_dir outputs/metrics \
    --limit 50 \
    --enable_fft --enable_sync

输出:
  - {out_dir}/fusion_method_comparison.json : 每个视频每种方法的结果
  - {out_dir}/fusion_method_comparison.csv  : 表格化结果
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List
import logging
import math

from .utils import setup_logger, load_json, save_json
from .predict_video import predict_video


def evaluate_video_methods(
    video_path: Path,
    model_path: Path = None,
    enable_fft: bool = False,
    enable_sync: bool = False,
    device=None,
    methods: List[str] = None,
):
    if methods is None:
        methods = ["average", "top_vote", "weighted_average"]

    per_method = {}
    for m in methods:
        try:
            res = predict_video(
                video_path,
                model_path=model_path,
                fusion_method=m,
                output_path=None,
                device=device,
                enable_fft=enable_fft,
                enable_sync=enable_sync,
            )

            # prefer calibrated label if present
            calibrated = res.get('calibrated')
            if calibrated:
                label = calibrated.get('calibrated_label')
                prob = calibrated.get('calibrated_fake_prob')
            else:
                label = res.get('video_label')
                prob = res.get('video_fake_prob')

            per_method[m] = {
                'video_label': res.get('video_label'),
                'video_fake_prob': res.get('video_fake_prob'),
                'calibrated_label': label,
                'calibrated_fake_prob': prob,
                'num_faces': res.get('num_faces'),
                'module_b': res.get('module_b', {}),
                'error': res.get('error') if res.get('error') else None,
            }

        except Exception as e:
            per_method[m] = {'error': str(e)}

    return per_method


def load_metadata(metadata_path: Path) -> Dict[str, int]:
    if not metadata_path.exists():
        logging.warning(f"metadata not found: {metadata_path}")
        return {}

    meta = load_json(metadata_path)
    # metadata.json might map video_id -> {label: 'FAKE'/'REAL'}; normalize to lowercase labels 'fake'/'real'
    out = {}
    from pathlib import Path as _P
    for vid, info in meta.items():
        stem = _P(vid).stem
        # some metadata entries are nested; try common keys
        label = None
        if isinstance(info, dict):
            label = info.get('label') or info.get('label_name') or info.get('video_label')
        else:
            label = info

        if label is None:
            continue

        lab = str(label).strip().lower()
        if lab.startswith('f'):
            out[stem] = 'fake'
        elif lab.startswith('r'):
            out[stem] = 'real'
        else:
            out[stem] = lab

    return out


def main():
    parser = argparse.ArgumentParser(description="批量重算并比较融合方法")
    parser.add_argument('--video_dir', type=str, default='data/kaggle_dfdc/00')
    parser.add_argument('--metadata', type=str, default='data/kaggle_dfdc/00/metadata.json')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='outputs/metrics')
    parser.add_argument('--limit', type=int, default=0, help='限制处理的视频数量 (0 表示全部)')
    parser.add_argument('--enable_fft', action='store_true')
    parser.add_argument('--enable_sync', action='store_true')

    args = parser.parse_args()

    setup_logger()

    video_dir = Path(args.video_dir)
    metadata_path = Path(args.metadata)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_path)

    # collect video files
    exts = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    videos = []
    for e in exts:
        videos.extend(sorted(video_dir.glob(e)))

    if args.limit and args.limit > 0:
        videos = videos[:args.limit]

    logging.info(f"重算 {len(videos)} 个视频 (limit={args.limit}) in {video_dir}")

    methods = ["average", "top_vote", "weighted_average"]

    records = []
    counts = {m: {'correct': 0, 'total': 0} for m in methods}

    for vid in videos:
        vid_stem = vid.stem
        true_label = metadata.get(vid_stem)

        per_method = evaluate_video_methods(
            vid,
            model_path=Path(args.model_path) if args.model_path else None,
            enable_fft=args.enable_fft,
            enable_sync=args.enable_sync,
            methods=methods,
        )

        for m in methods:
            entry = per_method.get(m, {})
            predicted = entry.get('calibrated_label') or entry.get('video_label')
            prob = entry.get('calibrated_fake_prob')

            correct = None
            if true_label:
                counts[m]['total'] += 1
                if predicted:
                    if predicted == true_label:
                        counts[m]['correct'] += 1
                        correct = True
                    else:
                        correct = False

            records.append({
                'video_id': vid_stem,
                'video_path': str(vid),
                'method': m,
                'predicted_label': predicted,
                'predicted_prob': float(prob) if prob is not None and not (isinstance(prob, str) and math.isnan(float('nan'))) else None,
                'true_label': true_label,
                'correct': correct,
                'num_faces': entry.get('num_faces'),
            })

    # compute accuracies
    summary = {}
    for m in methods:
        total = counts[m]['total']
        correct = counts[m]['correct']
        acc = float(correct) / total if total > 0 else None
        summary[m] = {
            'total_videos_with_label': total,
            'correct': correct,
            'accuracy': acc
        }

    out_json = out_dir / 'fusion_method_comparison.json'
    out_csv = out_dir / 'fusion_method_comparison.csv'

    save_json({'summary': summary, 'records': records}, out_json)

    # write CSV
    try:
        import csv
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['video_id', 'video_path', 'method', 'predicted_label', 'predicted_prob', 'true_label', 'correct', 'num_faces'])
            writer.writeheader()
            for r in records:
                writer.writerow(r)
    except Exception as e:
        logging.warning(f"写 CSV 失败: {e}")

    # print summary
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    logging.info(f"结果保存: {out_json}, {out_csv}")


if __name__ == '__main__':
    main()
