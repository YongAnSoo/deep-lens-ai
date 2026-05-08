"""
评估脚本
在测试集上评估训练好的模型
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from tqdm import tqdm
import numpy as np

from . import config
from .model import load_model_for_inference
from .utils import setup_logger, save_json, get_device, load_json


def get_test_transform(image_size: int = 224) -> transforms.Compose:
    """
    获取测试集数据转换

    Args:
        image_size: 图像大小

    Returns:
        transforms.Compose
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])

    return transform


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    评估模型

    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 计算设备

    Returns:
        (metrics, all_labels, all_preds, all_probs)
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    logging.info("开始评估...")

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # fake 类的概率

    # 转换为 numpy 数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1])
        }
    }

    return metrics, all_labels, all_preds, all_probs


def evaluate(
    model_path: Path = None,
    dataset_dir: Path = None,
    batch_size: int = 16,
    num_workers: int = 4
):
    """
    评估模型

    Args:
        model_path: 模型路径
        dataset_dir: 数据集目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
    """
    # 默认路径
    if model_path is None:
        model_path = config.BEST_MODEL_PATH

    if dataset_dir is None:
        dataset_dir = config.DATASET_DIR

    # 检查模型是否存在
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载模型配置
    model_config_path = config.MODEL_CONFIG_PATH
    if model_config_path.exists():
        model_config = load_json(model_config_path)
        model_name = model_config.get('model_name', config.MODEL_NAME)
        image_size = model_config.get('image_size', config.IMAGE_SIZE)
        logging.info(f"从配置加载: model_name={model_name}, image_size={image_size}")
    else:
        model_name = config.MODEL_NAME
        image_size = config.IMAGE_SIZE
        logging.warning(f"未找到模型配置，使用默认值: model_name={model_name}, image_size={image_size}")

    # 获取设备
    device = get_device()
    logging.info(f"使用设备: {device}")

    # 加载模型
    logging.info(f"加载模型: {model_path}")
    model, _ = load_model_for_inference(str(model_path), model_name=model_name, device=device)

    # 创建测试数据加载器
    test_dir = dataset_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"测试集目录不存在: {test_dir}")

    test_dataset = datasets.ImageFolder(test_dir, transform=get_test_transform(image_size))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logging.info(f"测试集大小: {len(test_dataset)}")
    logging.info(f"类别映射: {test_dataset.class_to_idx}")

    # 评估模型
    metrics, all_labels, all_preds, all_probs = evaluate_model(model, test_loader, device)

    # 打印结果
    logging.info("=" * 60)
    logging.info("测试集评估结果")
    logging.info("=" * 60)
    logging.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logging.info(f"Precision: {metrics['precision']:.4f}")
    logging.info(f"Recall:    {metrics['recall']:.4f}")
    logging.info(f"F1 Score:  {metrics['f1']:.4f}")
    logging.info(f"AUC:       {metrics['auc']:.4f}")
    logging.info("")
    logging.info("混淆矩阵:")
    logging.info(f"  TN: {metrics['confusion_matrix']['true_negative']:4d}  |  FP: {metrics['confusion_matrix']['false_positive']:4d}")
    logging.info(f"  FN: {metrics['confusion_matrix']['false_negative']:4d}  |  TP: {metrics['confusion_matrix']['true_positive']:4d}")
    logging.info("=" * 60)

    # 保存结果
    output_path = config.METRICS_DIR / "test_metrics.json"
    save_json(metrics, output_path)
    logging.info(f"评估结果已保存: {output_path}")

    return metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="评估 Deepfake 检测模型")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--dataset_dir", type=str, default=None, help="数据集目录")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")

    args = parser.parse_args()

    # 设置日志
    setup_logger()

    # 转换路径
    model_path = Path(args.model_path) if args.model_path else None
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None

    # 评估
    evaluate(
        model_path=model_path,
        dataset_dir=dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
