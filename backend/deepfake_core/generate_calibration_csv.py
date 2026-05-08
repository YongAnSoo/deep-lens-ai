"""
从 outputs/predictions 聚合预测结果并生成用于校准的 CSV

输出路径: outputs/metrics/calibration_dataset.csv

使用:
  python -m src.generate_calibration_csv --pred_dir outputs/predictions --metadata data/kaggle_dfdc/00/metadata.json --out outputs/metrics/calibration_dataset.csv

"""
from pathlib import Path
import json
import argparse
import logging
import csv

from .utils import setup_logger


def load_metadata(meta_path: Path) -> dict:
    if not meta_path.exists():
        logging.warning(f"metadata.json not found: {meta_path}")
        return {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def find_result_files(pred_dir: Path):
    files = list(pred_dir.rglob('*_result.json'))
    # also include top-level JSONs that may be summaries
    for p in pred_dir.glob('*.json'):
        if p.name.endswith('_result.json') or p.name in ('test_result.json', 'test_video_result.json', 'random_test_result.json'):
            if p not in files:
                files.append(p)
    return files


def extract_scores(result: dict):
    # video_fake_prob
    video_fake_prob = result.get('video_fake_prob')
    if video_fake_prob is None:
        # maybe only video_real_prob
        vr = result.get('video_real_prob')
        if vr is not None:
            video_fake_prob = 1.0 - float(vr)

    # Module B: try module_b then module_b_details
    frequency_score = None
    sync_score = None

    mb = result.get('module_b') or {}
    if mb:
        frequency_score = mb.get('frequency_score')
        sync_score = mb.get('sync_score')

    if frequency_score is None or sync_score is None:
        mbd = result.get('module_b_details') or {}
        fft = mbd.get('fft') or {}
        sync = mbd.get('sync') or {}
        if frequency_score is None:
            frequency_score = fft.get('frequency_score') or fft.get('score')
        if sync_score is None:
            sync_score = sync.get('sync_score') or sync.get('score')

    # Normalize to float or None
    try:
        video_fake_prob = float(video_fake_prob) if video_fake_prob is not None else None
    except Exception:
        video_fake_prob = None

    try:
        frequency_score = float(frequency_score) if frequency_score is not None else None
    except Exception:
        frequency_score = None

    try:
        sync_score = float(sync_score) if sync_score is not None else None
    except Exception:
        sync_score = None

    return video_fake_prob, frequency_score, sync_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, default='outputs/predictions')
    parser.add_argument('--metadata', type=str, default='data/kaggle_dfdc/00/metadata.json')
    parser.add_argument('--out', type=str, default='outputs/metrics/calibration_dataset.csv')

    args = parser.parse_args()

    setup_logger()

    pred_dir = Path(args.pred_dir)
    meta_path = Path(args.metadata)
    out_path = Path(args.out)

    metadata = load_metadata(meta_path)

    files = find_result_files(pred_dir)
    logging.info(f"Found {len(files)} result files")

    rows = []
    for p in files:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                res = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load {p}: {e}")
            continue

        # Infer video id: try result['video_path'] else filename stem + .mp4
        video_path = res.get('video_path')
        if video_path:
            vid = Path(video_path).name
        else:
            stem = p.stem
            # remove trailing _result if present
            if stem.endswith('_result'):
                stem = stem[:-7]
            vid = stem + '.mp4'

        # ground-truth from metadata
        meta_entry = metadata.get(vid)
        if meta_entry and 'label' in meta_entry:
            label = meta_entry['label'].lower()
        else:
            # fallback: infer from directory name
            if 'fake' in p.parts:
                label = 'fake'
            elif 'real' in p.parts:
                label = 'real'
            else:
                label = ''

        video_fake_prob, frequency_score, sync_score = extract_scores(res)

        rows.append({
            'video_id': vid,
            'video_fake_prob': video_fake_prob,
            'frequency_score': frequency_score,
            'sync_score': sync_score,
            'label': label,
            'source_json': str(p)
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['video_id', 'video_fake_prob', 'frequency_score', 'sync_score', 'label', 'source_json']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    logging.info(f"Calibration CSV saved to: {out_path} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
