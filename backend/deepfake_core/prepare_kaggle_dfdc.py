"""
准备 Kaggle DFDC 数据集
读取 metadata.json 并将视频按 REAL/FAKE 分类
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

from . import config
from . import utils


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="准备 Kaggle DFDC 数据集")
    parser.add_argument(
        "--chunk_dir",
        type=str,
        default=str(config.KAGGLE_DFDC_DIR),
        help="Kaggle DFDC 数据块目录路径"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["symlink", "copy"],
        default="symlink",
        help="文件操作模式：symlink（符号链接）或 copy（复制）"
    )
    parser.add_argument(
        "--max_real",
        type=int,
        default=None,
        help="最多处理的真实视频数量（None 表示全部）"
    )
    parser.add_argument(
        "--max_fake",
        type=int,
        default=None,
        help="最多处理的假视频数量（None 表示全部）"
    )
    return parser.parse_args()


def load_metadata(chunk_dir: Path) -> Dict:
    """
    加载 metadata.json 文件

    Args:
        chunk_dir: 数据块目录路径

    Returns:
        元数据字典
    """
    metadata_path = chunk_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.json 不存在: {metadata_path}\n"
            f"请确保已将 Kaggle DFDC 数据解压到: {chunk_dir}"
        )

    metadata = utils.load_metadata_json(metadata_path)
    logging.info(f"加载了 {len(metadata)} 个视频的元数据")

    return metadata


def prepare_videos(
    chunk_dir: Path,
    metadata: Dict,
    mode: str = "symlink",
    max_real: int = None,
    max_fake: int = None
) -> List[Dict]:
    """
    根据 metadata 将视频分类到 real/fake 文件夹

    Args:
        chunk_dir: 数据块目录路径
        metadata: 元数据字典
        mode: 操作模式（symlink 或 copy）
        max_real: 最多处理的真实视频数量
        max_fake: 最多处理的假视频数量

    Returns:
        处理记录列表
    """
    records = []
    real_count = 0
    fake_count = 0

    # 确保目标目录存在
    real_dir = config.RAW_VIDEOS_DIR / "real"
    fake_dir = config.RAW_VIDEOS_DIR / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"开始处理视频，模式: {mode}")

    for filename, info in tqdm(metadata.items(), desc="处理视频"):
        # 检查视频文件是否存在
        source_path = chunk_dir / filename

        if not source_path.exists():
            logging.warning(f"视频文件不存在，跳过: {filename}")
            records.append({
                "filename": filename,
                "source_path": str(source_path),
                "target_path": None,
                "label": info.get("label", "UNKNOWN"),
                "original": info.get("original", None),
                "split": info.get("split", None),
                "exists": False,
                "action_taken": "skipped_not_found"
            })
            continue

        # 获取标签
        label = info.get("label", "").upper()

        if label not in ["REAL", "FAKE"]:
            logging.warning(f"未知标签 '{label}'，跳过: {filename}")
            records.append({
                "filename": filename,
                "source_path": str(source_path),
                "target_path": None,
                "label": label,
                "original": info.get("original", None),
                "split": info.get("split", None),
                "exists": True,
                "action_taken": "skipped_unknown_label"
            })
            continue

        # 检查是否达到数量限制
        if label == "REAL":
            if max_real is not None and real_count >= max_real:
                continue
            target_dir = real_dir
        else:  # FAKE
            if max_fake is not None and fake_count >= max_fake:
                continue
            target_dir = fake_dir

        # 目标路径
        target_path = target_dir / filename

        # 执行符号链接或复制
        action = utils.safe_symlink_or_copy(source_path, target_path, mode)

        # 更新计数
        if label == "REAL":
            real_count += 1
        else:
            fake_count += 1

        # 记录
        records.append({
            "filename": filename,
            "source_path": str(source_path),
            "target_path": str(target_path),
            "label": label,
            "original": info.get("original", None),
            "split": info.get("split", None),
            "exists": True,
            "action_taken": action
        })

    logging.info(f"处理完成: {real_count} 个真实视频, {fake_count} 个假视频")

    return records


def save_records(records: List[Dict], output_path: Path):
    """
    保存处理记录到 CSV 文件

    Args:
        records: 处理记录列表
        output_path: 输出 CSV 路径
    """
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')

    logging.info(f"已保存视频索引: {output_path}")

    # 打印统计信息
    if len(df) > 0:
        logging.info("\n=== 统计信息 ===")
        logging.info(f"总视频数: {len(df)}")
        logging.info(f"真实视频: {len(df[df['label'] == 'REAL'])}")
        logging.info(f"假视频: {len(df[df['label'] == 'FAKE'])}")
        logging.info(f"符号链接: {len(df[df['action_taken'] == 'symlink'])}")
        logging.info(f"复制: {len(df[df['action_taken'] == 'copy'])}")
        logging.info(f"跳过: {len(df[df['action_taken'].str.startswith('skipped')])}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置日志
    utils.setup_logger()

    # 设置随机种子
    utils.set_random_seed(config.RANDOM_SEED)

    # 创建必要的目录
    utils.create_directories()

    logging.info("=" * 60)
    logging.info("准备 Kaggle DFDC 数据集")
    logging.info("=" * 60)
    logging.info(f"数据块目录: {args.chunk_dir}")
    logging.info(f"操作模式: {args.mode}")
    logging.info(f"最大真实视频数: {args.max_real if args.max_real else '无限制'}")
    logging.info(f"最大假视频数: {args.max_fake if args.max_fake else '无限制'}")

    # 转换路径
    chunk_dir = Path(args.chunk_dir)

    # 加载 metadata
    metadata = load_metadata(chunk_dir)

    # 处理视频
    records = prepare_videos(
        chunk_dir=chunk_dir,
        metadata=metadata,
        mode=args.mode,
        max_real=args.max_real,
        max_fake=args.max_fake
    )

    # 保存记录
    output_path = config.METRICS_DIR / "kaggle_video_index.csv"
    save_records(records, output_path)

    logging.info("=" * 60)
    logging.info("数据准备完成！")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
