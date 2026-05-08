"""
从视频中提取帧
从 raw_videos 目录读取视频，按指定间隔提取帧并保存
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict
import cv2
import pandas as pd
from tqdm import tqdm

from . import config
from . import utils


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="从视频中提取帧")
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=config.FRAME_INTERVAL,
        help=f"帧提取间隔（默认: {config.FRAME_INTERVAL}）"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=config.MAX_FRAMES_PER_VIDEO,
        help=f"每个视频最多提取的帧数（默认: {config.MAX_FRAMES_PER_VIDEO}）"
    )
    return parser.parse_args()


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    label: str,
    frame_interval: int = 10,
    max_frames: int = 30
) -> List[Dict]:
    """
    从单个视频中提取帧

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        label: 标签（real 或 fake）
        frame_interval: 帧提取间隔
        max_frames: 最多提取的帧数

    Returns:
        帧元数据列表
    """
    records = []
    video_name = video_path.stem

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logging.warning(f"无法打开视频: {video_path}")
        return records

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    extracted_count = 0

    while cap.isOpened() and extracted_count < max_frames:
        ret, frame = cap.read()

        if not ret:
            break

        # 按间隔提取帧
        if frame_count % frame_interval == 0:
            # 生成帧文件名: label__videoname__frame000001.jpg
            frame_filename = f"{label}__{video_name}__frame{extracted_count+1:06d}.jpg"
            frame_path = output_dir / frame_filename

            # 保存帧
            cv2.imwrite(str(frame_path), frame)

            # 计算时间戳
            timestamp_sec = frame_count / fps if fps > 0 else 0

            # 记录元数据
            records.append({
                "video_filename": video_path.name,
                "label": label,
                "frame_path": str(frame_path),
                "frame_index": frame_count,
                "timestamp_sec": timestamp_sec
            })

            extracted_count += 1

        frame_count += 1

    cap.release()

    return records


def extract_frames_from_directory(
    video_dir: Path,
    output_dir: Path,
    label: str,
    frame_interval: int = 10,
    max_frames: int = 30
) -> List[Dict]:
    """
    从目录中的所有视频提取帧

    Args:
        video_dir: 视频目录
        output_dir: 输出目录
        label: 标签（real 或 fake）
        frame_interval: 帧提取间隔
        max_frames: 每个视频最多提取的帧数

    Returns:
        所有帧的元数据列表
    """
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有视频文件
    video_files = list(video_dir.glob("*.mp4"))

    if not video_files:
        logging.warning(f"目录中没有找到视频文件: {video_dir}")
        return []

    logging.info(f"在 {video_dir} 中找到 {len(video_files)} 个视频")

    all_records = []

    for video_path in tqdm(video_files, desc=f"提取 {label} 帧"):
        records = extract_frames_from_video(
            video_path=video_path,
            output_dir=output_dir,
            label=label,
            frame_interval=frame_interval,
            max_frames=max_frames
        )
        all_records.extend(records)

    logging.info(f"从 {len(video_files)} 个 {label} 视频中提取了 {len(all_records)} 帧")

    return all_records


def save_metadata(records: List[Dict], output_path: Path):
    """
    保存帧元数据到 CSV

    Args:
        records: 元数据记录列表
        output_path: 输出 CSV 路径
    """
    if not records:
        logging.warning("没有帧元数据可保存")
        return

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')

    logging.info(f"已保存帧元数据: {output_path}")

    # 打印统计信息
    logging.info("\n=== 统计信息 ===")
    logging.info(f"总帧数: {len(df)}")
    logging.info(f"真实帧数: {len(df[df['label'] == 'real'])}")
    logging.info(f"假帧数: {len(df[df['label'] == 'fake'])}")
    logging.info(f"唯一视频数: {df['video_filename'].nunique()}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置日志
    utils.setup_logger()

    logging.info("=" * 60)
    logging.info("从视频中提取帧")
    logging.info("=" * 60)
    logging.info(f"帧提取间隔: {args.frame_interval}")
    logging.info(f"每个视频最多提取帧数: {args.max_frames}")

    all_records = []

    # 提取真实视频的帧
    real_video_dir = config.RAW_VIDEOS_DIR / "real"
    real_output_dir = config.FRAMES_DIR / "real"

    if real_video_dir.exists():
        real_records = extract_frames_from_directory(
            video_dir=real_video_dir,
            output_dir=real_output_dir,
            label="real",
            frame_interval=args.frame_interval,
            max_frames=args.max_frames
        )
        all_records.extend(real_records)
    else:
        logging.warning(f"真实视频目录不存在: {real_video_dir}")

    # 提取假视频的帧
    fake_video_dir = config.RAW_VIDEOS_DIR / "fake"
    fake_output_dir = config.FRAMES_DIR / "fake"

    if fake_video_dir.exists():
        fake_records = extract_frames_from_directory(
            video_dir=fake_video_dir,
            output_dir=fake_output_dir,
            label="fake",
            frame_interval=args.frame_interval,
            max_frames=args.max_frames
        )
        all_records.extend(fake_records)
    else:
        logging.warning(f"假视频目录不存在: {fake_video_dir}")

    # 保存元数据
    output_path = config.METRICS_DIR / "frame_metadata.csv"
    save_metadata(all_records, output_path)

    logging.info("=" * 60)
    logging.info("帧提取完成！")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
