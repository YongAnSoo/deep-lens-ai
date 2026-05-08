"""
通用工具函数
包含目录创建、JSON操作、日志记录等功能
"""

import json
import random
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Union
import numpy as np
import torch


# ============================================================================
# 目录管理
# ============================================================================
def create_directories():
    """
    创建项目所需的所有目录
    """
    from . import config

    directories = [
        config.DATA_DIR,
        config.KAGGLE_DFDC_DIR,
        config.RAW_VIDEOS_DIR / "real",
        config.RAW_VIDEOS_DIR / "fake",
        config.FRAMES_DIR / "real",
        config.FRAMES_DIR / "fake",
        config.FACES_DIR / "real",
        config.FACES_DIR / "fake",
        config.DATASET_DIR / "train" / "real",
        config.DATASET_DIR / "train" / "fake",
        config.DATASET_DIR / "val" / "real",
        config.DATASET_DIR / "val" / "fake",
        config.DATASET_DIR / "test" / "real",
        config.DATASET_DIR / "test" / "fake",
        config.MODELS_DIR,
        config.GRADCAM_DIR,
        config.PREDICTIONS_DIR,
        config.METRICS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    logging.info(f"已创建 {len(directories)} 个目录")


# ============================================================================
# JSON 操作
# ============================================================================
def save_json(data: Union[Dict, List], filepath: Union[str, Path], indent: int = 2):
    """
    保存数据为 JSON 文件

    Args:
        data: 要保存的数据（字典或列表）
        filepath: 保存路径
        indent: JSON 缩进空格数
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    logging.info(f"已保存 JSON 文件: {filepath}")


def load_json(filepath: Union[str, Path]) -> Union[Dict, List]:
    """
    加载 JSON 文件

    Args:
        filepath: JSON 文件路径

    Returns:
        解析后的数据（字典或列表）
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logging.info(f"已加载 JSON 文件: {filepath}")
    return data


def load_metadata_json(filepath: Union[str, Path]) -> Dict:
    """
    加载 Kaggle DFDC metadata.json 文件
    处理可能的格式问题

    Args:
        filepath: metadata.json 文件路径

    Returns:
        元数据字典
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"metadata.json 文件不存在: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logging.info(f"已加载 metadata.json: {filepath}, 包含 {len(metadata)} 个视频")
        return metadata
    except json.JSONDecodeError as e:
        logging.error(f"解析 metadata.json 失败: {e}")
        raise


# ============================================================================
# 随机种子设置
# ============================================================================
def set_random_seed(seed: int = 42):
    """
    设置所有随机种子以确保可重复性

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logging.info(f"已设置随机种子: {seed}")


# ============================================================================
# 设备管理
# ============================================================================
def get_device() -> torch.device:
    """
    获取可用的最佳计算设备

    Returns:
        torch.device 对象
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        logging.info(f"使用 CUDA 设备: {device_name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("使用 MPS 设备 (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logging.info("使用 CPU 设备")

    return device


# ============================================================================
# 时间戳格式化
# ============================================================================
def format_timestamp(seconds: float) -> str:
    """
    将秒数格式化为 HH:MM:SS.mmm

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def get_timestamp() -> str:
    """
    获取当前时间戳字符串

    Returns:
        格式化的时间戳 (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# 文件操作
# ============================================================================
def safe_symlink_or_copy(src: Path, dst: Path, mode: str = "symlink") -> str:
    """
    安全地创建符号链接或复制文件

    Args:
        src: 源文件路径
        dst: 目标文件路径
        mode: 操作模式 ("symlink" 或 "copy")

    Returns:
        执行的操作 ("symlink", "copy", 或 "skipped")
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    # 如果目标已存在，跳过
    if dst.exists():
        return "skipped"

    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return "symlink"
        except (OSError, NotImplementedError):
            # 符号链接失败，回退到复制
            logging.warning(f"符号链接失败，回退到复制: {src.name}")
            shutil.copy2(src, dst)
            return "copy"
    else:
        shutil.copy2(src, dst)
        return "copy"


# ============================================================================
# 日志配置
# ============================================================================
def setup_logger(
    name: str = "deepfake_module_a",
    level: int = logging.INFO,
    log_file: Union[str, Path] = None
) -> logging.Logger:
    """
    配置日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 可选的日志文件路径

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器（可选）
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================================
# 数据统计
# ============================================================================
def count_files_in_directory(directory: Path, pattern: str = "*") -> int:
    """
    统计目录中的文件数量

    Args:
        directory: 目录路径
        pattern: 文件匹配模式

    Returns:
        文件数量
    """
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def get_video_filename_without_ext(video_path: Union[str, Path]) -> str:
    """
    获取视频文件名（不含扩展名）

    Args:
        video_path: 视频文件路径

    Returns:
        文件名（不含扩展名）
    """
    return Path(video_path).stem


# ============================================================================
# 模型相关
# ============================================================================
def count_parameters(model: torch.nn.Module) -> int:
    """
    统计模型的可训练参数数量

    Args:
        model: PyTorch 模型

    Returns:
        可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: Union[str, Path]
):
    """
    保存模型检查点

    Args:
        model: PyTorch 模型
        optimizer: 优化器
        epoch: 当前 epoch
        metrics: 评估指标字典
        filepath: 保存路径
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }

    torch.save(checkpoint, filepath)
    logging.info(f"已保存检查点: {filepath}")


def load_checkpoint(
    filepath: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None
) -> Dict:
    """
    加载模型检查点

    Args:
        filepath: 检查点文件路径
        model: PyTorch 模型
        optimizer: 可选的优化器

    Returns:
        检查点字典
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"检查点文件不存在: {filepath}")

    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logging.info(f"已加载检查点: {filepath}")
    return checkpoint
