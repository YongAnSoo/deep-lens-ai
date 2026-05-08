"""
训练脚本
训练 EfficientNet 模型进行 Deepfake 检测
"""

import os
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Windows + mixed scientific stack may load duplicate OpenMP runtimes.
# This keeps training launch stable in common conda/pip mixed envs.
if os.name == "nt" and "KMP_DUPLICATE_LIB_OK" not in os.environ:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import pandas as pd

from . import config
from .model import get_model
from .utils import setup_logger, save_json, set_random_seed, get_device


def get_transforms(image_size: int = 224, augment: bool = True) -> transforms.Compose:
    """
    获取数据转换

    Args:
        image_size: 图像大小
        augment: 是否使用数据增强

    Returns:
        transforms.Compose
    """
    if augment:
        # 训练集：更强的数据增强以减少过拟合
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])
    else:
        # 验证集/测试集：仅标准化
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])

    return transform


def get_dataloaders(
    dataset_dir: Path,
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器

    Args:
        dataset_dir: 数据集根目录
        batch_size: 批次大小
        image_size: 图像大小
        num_workers: 数据加载线程数

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    test_dir = dataset_dir / "test"

    # 检查目录是否存在
    for split_dir in [train_dir, val_dir, test_dir]:
        if not split_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {split_dir}")

    # 创建数据集
    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(image_size, augment=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms(image_size, augment=False))
    test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms(image_size, augment=False))

    logging.info(f"训练集大小: {len(train_dataset)}")
    logging.info(f"验证集大小: {len(val_dataset)}")
    logging.info(f"测试集大小: {len(test_dataset)}")
    logging.info(f"类别映射: {train_dataset.class_to_idx}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler = None
) -> Dict[str, float]:
    """
    训练一个 epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        epoch: 当前 epoch

    Returns:
        训练指标字典
    """
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    use_amp = scaler is not None and scaler.is_enabled()

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 前向传播
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # 反向传播
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 统计
        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())  # fake 类的概率

        # 更新进度条
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    # 计算指标
    avg_loss = running_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

    return metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler = None
) -> Dict[str, float]:
    """
    验证模型

    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
        epoch: 当前 epoch

    Returns:
        验证指标字典
    """
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    use_amp = scaler is not None and scaler.is_enabled()

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 前向传播
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 统计
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    # 计算指标
    avg_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

    return metrics


def train(
    model_name: str = "efficientnet_b0",
    batch_size: int = 16,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    patience: int = 5,
    image_size: int = 224,
    num_workers: int = 4,
    use_amp: bool = True
):
    """
    训练模型

    Args:
        model_name: 模型名称
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        patience: 早停耐心值
        image_size: 图像大小
        num_workers: 数据加载线程数
    """
    # 设置随机种子
    set_random_seed(config.RANDOM_SEED)

    # 获取设备
    device = get_device()
    logging.info(f"使用设备: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # 创建数据加载器
    logging.info("加载数据集...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config.DATASET_DIR,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers
    )

    # 创建模型
    logging.info(f"创建模型: {model_name}")
    model = get_model(model_name=model_name, num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2
    )

    # 训练循环
    best_metric = 0.0
    best_epoch = 0
    patience_counter = 0
    training_log = []

    logging.info("开始训练...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # 训练
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler)

        # 验证
        val_metrics = validate(model, val_loader, criterion, device, epoch, scaler)

        epoch_time = time.time() - epoch_start_time

        # 记录日志
        log_entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "epoch_time": epoch_time,
            "lr": optimizer.param_groups[0]['lr']
        }
        training_log.append(log_entry)

        logging.info(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Train F1: {train_metrics['f1']:.4f}, "
            f"Train AUC: {train_metrics['auc']:.4f}"
        )
        logging.info(
            f"Epoch {epoch}/{epochs} - "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Val F1: {val_metrics['f1']:.4f}, "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )

        # 选择最佳指标（优先 AUC，如果 AUC 为 0 则使用 F1）
        current_metric = val_metrics["auc"] if val_metrics["auc"] > 0 else val_metrics["f1"]
        metric_name = "AUC" if val_metrics["auc"] > 0 else "F1"

        # 更新学习率
        scheduler.step(current_metric)

        # 保存最佳模型
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0

            # 保存模型
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'metric_name': metric_name,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': {
                    'model_name': model_name,
                    'num_classes': config.NUM_CLASSES,
                    'image_size': image_size,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate
                }
            }

            torch.save(checkpoint, config.BEST_MODEL_PATH)
            logging.info(f"保存最佳模型 (Epoch {epoch}, {metric_name}: {best_metric:.4f})")

            # 保存模型配置
            model_config = {
                'model_name': model_name,
                'num_classes': config.NUM_CLASSES,
                'image_size': image_size,
                'class_names': config.CLASS_NAMES,
                'best_epoch': best_epoch,
                'best_metric': best_metric,
                'metric_name': metric_name
            }
            save_json(model_config, config.MODEL_CONFIG_PATH)
        else:
            patience_counter += 1
            logging.info(f"未改进 ({patience_counter}/{patience})")

        # 早停
        if patience_counter >= patience:
            logging.info(f"早停触发，最佳 {metric_name}: {best_metric:.4f} (Epoch {best_epoch})")
            break

    total_time = time.time() - start_time
    logging.info(f"训练完成，总时间: {total_time:.2f} 秒")

    # 保存训练日志
    log_df = pd.DataFrame(training_log)
    log_csv_path = config.METRICS_DIR / "training_log.csv"
    log_df.to_csv(log_csv_path, index=False)
    logging.info(f"训练日志已保存: {log_csv_path}")

    # 保存训练摘要
    summary = {
        'model_name': model_name,
        'total_epochs': epoch,
        'best_epoch': best_epoch,
        'best_metric': best_metric,
        'metric_name': metric_name,
        'total_time_seconds': total_time,
        'final_train_metrics': train_metrics,
        'final_val_metrics': val_metrics,
        'config': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'image_size': image_size,
            'patience': patience
        }
    }

    summary_path = config.METRICS_DIR / "train_summary.json"
    save_json(summary, summary_path)
    logging.info(f"训练摘要已保存: {summary_path}")

    return model, training_log


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练 Deepfake 检测模型")
    parser.add_argument("--model_name", type=str, default=config.MODEL_NAME, help="模型名称")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="批次大小")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="训练轮数")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY, help="权重衰减")
    parser.add_argument("--patience", type=int, default=config.PATIENCE, help="早停耐心值")
    parser.add_argument("--image_size", type=int, default=config.IMAGE_SIZE, help="图像大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--no_amp", action="store_true", help="禁用混合精度训练")

    args = parser.parse_args()

    # 设置日志
    setup_logger()

    # 训练
    train(
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        image_size=args.image_size,
        num_workers=args.num_workers,
        use_amp=not args.no_amp
    )


if __name__ == "__main__":
    main()
