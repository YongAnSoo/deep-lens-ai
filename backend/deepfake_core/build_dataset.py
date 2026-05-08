"""
构建训练数据集
将裁剪的人脸按视频分组，划分为 train/val/test 数据集
避免数据泄漏：同一视频的所有帧必须在同一个数据集中
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np

from . import config
from . import utils


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="构建训练数据集")
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=config.TRAIN_RATIO,
        help=f"训练集比例（默认: {config.TRAIN_RATIO}）"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=config.VAL_RATIO,
        help=f"验证集比例（默认: {config.VAL_RATIO}）"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=config.TEST_RATIO,
        help=f"测试集比例（默认: {config.TEST_RATIO}）"
    )
    return parser.parse_args()


def extract_video_name_from_face_path(face_path: Path) -> str:
    """
    从人脸文件名中提取原始视频名称

    文件名格式: label__videoname__frame000001.jpg

    Args:
        face_path: 人脸文件路径

    Returns:
        视频名称
    """
    filename = face_path.stem  # 去掉扩展名
    parts = filename.split("__")

    if len(parts) >= 2:
        return parts[1]  # videoname
    else:
        logging.warning(f"无法从文件名中提取视频名称: {filename}")
        return filename


def group_faces_by_video(faces_dir: Path, label: str) -> Dict[str, List[Path]]:
    """
    将人脸按原始视频分组

    Args:
        faces_dir: 人脸目录
        label: 标签（real 或 fake）

    Returns:
        字典：{video_name: [face_paths]}
    """
    video_groups = defaultdict(list)

    face_files = list(faces_dir.glob("*.jpg"))

    for face_path in face_files:
        video_name = extract_video_name_from_face_path(face_path)
        video_groups[video_name].append(face_path)

    logging.info(f"{label}: 找到 {len(face_files)} 张人脸，来自 {len(video_groups)} 个视频")

    return dict(video_groups)


def split_videos(
    video_groups: Dict[str, List[Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    将视频划分为 train/val/test

    Args:
        video_groups: 视频分组字典
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子

    Returns:
        (train_videos, val_videos, test_videos)
    """
    # 检查比例总和
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"比例总和必须为 1.0，当前为 {total_ratio}")

    # 获取所有视频名称
    video_names = list(video_groups.keys())

    # 随机打乱
    np.random.seed(random_seed)
    np.random.shuffle(video_names)

    # 计算划分点
    n_videos = len(video_names)
    n_train = int(n_videos * train_ratio)
    n_val = int(n_videos * val_ratio)

    # 划分
    train_videos = video_names[:n_train]
    val_videos = video_names[n_train:n_train + n_val]
    test_videos = video_names[n_train + n_val:]

    logging.info(f"视频划分: train={len(train_videos)}, val={len(val_videos)}, test={len(test_videos)}")

    return train_videos, val_videos, test_videos


def copy_faces_to_split(
    video_groups: Dict[str, List[Path]],
    video_list: List[str],
    target_dir: Path,
    label: str
) -> List[Dict]:
    """
    将指定视频的人脸复制到目标目录

    Args:
        video_groups: 视频分组字典
        video_list: 要复制的视频列表
        target_dir: 目标目录
        label: 标签（real 或 fake）

    Returns:
        复制记录列表
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    records = []
    copied_count = 0

    for video_name in video_list:
        if video_name not in video_groups:
            logging.warning(f"视频 {video_name} 不在分组中")
            continue

        face_paths = video_groups[video_name]

        for face_path in face_paths:
            target_path = target_dir / face_path.name

            # 复制文件
            shutil.copy2(face_path, target_path)
            copied_count += 1

            records.append({
                "video_name": video_name,
                "source_path": str(face_path),
                "target_path": str(target_path),
                "label": label
            })

    logging.info(f"复制了 {copied_count} 张 {label} 人脸到 {target_dir}")

    return records


def build_dataset(
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict:
    """
    构建完整的数据集

    Args:
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子

    Returns:
        统计信息字典
    """
    all_records = []
    stats = {
        "train": {"real": 0, "fake": 0},
        "val": {"real": 0, "fake": 0},
        "test": {"real": 0, "fake": 0}
    }

    # 处理真实人脸
    real_faces_dir = config.FACES_DIR / "real"
    if real_faces_dir.exists():
        real_video_groups = group_faces_by_video(real_faces_dir, "real")

        train_videos, val_videos, test_videos = split_videos(
            real_video_groups,
            train_ratio,
            val_ratio,
            test_ratio,
            random_seed
        )

        # 复制到各个数据集
        train_records = copy_faces_to_split(
            real_video_groups,
            train_videos,
            config.DATASET_DIR / "train" / "real",
            "real"
        )
        all_records.extend(train_records)
        stats["train"]["real"] = len(train_records)

        val_records = copy_faces_to_split(
            real_video_groups,
            val_videos,
            config.DATASET_DIR / "val" / "real",
            "real"
        )
        all_records.extend(val_records)
        stats["val"]["real"] = len(val_records)

        test_records = copy_faces_to_split(
            real_video_groups,
            test_videos,
            config.DATASET_DIR / "test" / "real",
            "real"
        )
        all_records.extend(test_records)
        stats["test"]["real"] = len(test_records)
    else:
        logging.warning(f"真实人脸目录不存在: {real_faces_dir}")

    # 处理假人脸
    fake_faces_dir = config.FACES_DIR / "fake"
    if fake_faces_dir.exists():
        fake_video_groups = group_faces_by_video(fake_faces_dir, "fake")

        train_videos, val_videos, test_videos = split_videos(
            fake_video_groups,
            train_ratio,
            val_ratio,
            test_ratio,
            random_seed
        )

        # 复制到各个数据集
        train_records = copy_faces_to_split(
            fake_video_groups,
            train_videos,
            config.DATASET_DIR / "train" / "fake",
            "fake"
        )
        all_records.extend(train_records)
        stats["train"]["fake"] = len(train_records)

        val_records = copy_faces_to_split(
            fake_video_groups,
            val_videos,
            config.DATASET_DIR / "val" / "fake",
            "fake"
        )
        all_records.extend(val_records)
        stats["val"]["fake"] = len(val_records)

        test_records = copy_faces_to_split(
            fake_video_groups,
            test_videos,
            config.DATASET_DIR / "test" / "fake",
            "fake"
        )
        all_records.extend(test_records)
        stats["test"]["fake"] = len(test_records)
    else:
        logging.warning(f"假人脸目录不存在: {fake_faces_dir}")

    return all_records, stats


def save_records(records: List[Dict], output_path: Path):
    """
    保存数据集记录到 CSV

    Args:
        records: 记录列表
        output_path: 输出 CSV 路径
    """
    if not records:
        logging.warning("没有记录可保存")
        return

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')

    logging.info(f"已保存数据集记录: {output_path}")


def save_stats(stats: Dict, output_path: Path):
    """
    保存统计信息到 JSON

    Args:
        stats: 统计信息字典
        output_path: 输出 JSON 路径
    """
    # 计算总数
    stats["total"] = {
        "train": stats["train"]["real"] + stats["train"]["fake"],
        "val": stats["val"]["real"] + stats["val"]["fake"],
        "test": stats["test"]["real"] + stats["test"]["fake"]
    }

    stats["total"]["all"] = stats["total"]["train"] + stats["total"]["val"] + stats["total"]["test"]

    utils.save_json(stats, output_path)

    # 打印统计信息
    logging.info("\n=== 数据集统计 ===")
    logging.info(f"训练集: {stats['total']['train']} (real: {stats['train']['real']}, fake: {stats['train']['fake']})")
    logging.info(f"验证集: {stats['total']['val']} (real: {stats['val']['real']}, fake: {stats['val']['fake']})")
    logging.info(f"测试集: {stats['total']['test']} (real: {stats['test']['real']}, fake: {stats['test']['fake']})")
    logging.info(f"总计: {stats['total']['all']}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置日志
    utils.setup_logger()

    # 设置随机种子
    utils.set_random_seed(config.RANDOM_SEED)

    logging.info("=" * 60)
    logging.info("构建训练数据集")
    logging.info("=" * 60)
    logging.info(f"训练集比例: {args.train_ratio}")
    logging.info(f"验证集比例: {args.val_ratio}")
    logging.info(f"测试集比例: {args.test_ratio}")

    # 构建数据集
    records, stats = build_dataset(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=config.RANDOM_SEED
    )

    # 保存记录
    records_path = config.METRICS_DIR / "split_faces.csv"
    save_records(records, records_path)

    # 保存统计信息
    stats_path = config.METRICS_DIR / "split_metadata.json"
    save_stats(stats, stats_path)

    logging.info("=" * 60)
    logging.info("数据集构建完成！")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
