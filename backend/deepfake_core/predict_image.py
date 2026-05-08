"""
单图预测脚本
对单张裁剪好的人脸图像进行预测
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
from torchvision import transforms
from PIL import Image

from . import config
from .model import load_model_for_inference
from .utils import setup_logger, get_device, load_json


def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    """
    获取推理数据转换

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


def predict_image(
    image_path: Path,
    model_path: Path = None,
    device: torch.device = None
) -> Dict:
    """
    预测单张图像

    Args:
        image_path: 图像路径
        model_path: 模型路径
        device: 计算设备

    Returns:
        预测结果字典
    """
    # 默认路径
    if model_path is None:
        model_path = config.BEST_MODEL_PATH

    if device is None:
        device = get_device()

    # 检查文件是否存在
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载模型配置
    model_config_path = config.MODEL_CONFIG_PATH
    if model_config_path.exists():
        model_config = load_json(model_config_path)
        model_name = model_config.get('model_name', config.MODEL_NAME)
        image_size = model_config.get('image_size', config.IMAGE_SIZE)
    else:
        model_name = config.MODEL_NAME
        image_size = config.IMAGE_SIZE
        logging.warning(f"未找到模型配置，使用默认值")

    # 加载模型
    logging.info(f"加载模型: {model_path}")
    model, _ = load_model_for_inference(str(model_path), model_name=model_name, device=device)

    # 加载图像
    logging.info(f"加载图像: {image_path}")
    image = Image.open(image_path).convert('RGB')

    # 预处理
    transform = get_inference_transform(image_size)
    image_tensor = transform(image).unsqueeze(0)  # 添加 batch 维度
    image_tensor = image_tensor.to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    # 提取概率
    real_prob = probs[0, config.REAL_CLASS].item()
    fake_prob = probs[0, config.FAKE_CLASS].item()

    # 构建结果
    result = {
        "image_path": str(image_path),
        "real_probability": float(real_prob),
        "fake_probability": float(fake_prob),
        "predicted_label": config.CLASS_NAMES[pred_class],
        "predicted_class": int(pred_class)
    }

    return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="预测单张人脸图像")
    parser.add_argument("--image", type=str, required=True, help="图像路径")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")

    args = parser.parse_args()

    # 设置日志
    setup_logger()

    # 转换路径
    image_path = Path(args.image)
    model_path = Path(args.model_path) if args.model_path else None

    # 预测
    result = predict_image(image_path, model_path)

    # 打印结果
    logging.info("=" * 60)
    logging.info("预测结果")
    logging.info("=" * 60)
    logging.info(f"图像路径:       {result['image_path']}")
    logging.info(f"Real 概率:      {result['real_probability']:.4f}")
    logging.info(f"Fake 概率:      {result['fake_probability']:.4f}")
    logging.info(f"预测标签:       {result['predicted_label']}")
    logging.info("=" * 60)

    return result


if __name__ == "__main__":
    main()
