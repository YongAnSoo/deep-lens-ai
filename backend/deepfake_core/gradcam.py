"""
Grad-CAM 可视化
生成 Grad-CAM 热力图，用于解释模型预测
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from . import config
from .model import load_model_for_inference
from .utils import setup_logger, get_device, load_json


class GradCAM:
    """
    Grad-CAM 实现
    适用于 EfficientNet
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        初始化 Grad-CAM

        Args:
            model: 模型
            target_layer: 目标层（通常是最后一个卷积层）
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """保存前向传播的激活值"""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """保存反向传播的梯度"""
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None
    ) -> np.ndarray:
        """
        生成 CAM 热力图

        Args:
            input_tensor: 输入张量 (1, 3, H, W)
            target_class: 目标类别（None 表示预测类别）

        Returns:
            CAM 热力图 (H, W)
        """
        # 前向传播
        self.model.eval()
        output = self.model(input_tensor)

        # 如果未指定目标类别，使用预测类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 反向传播
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # 计算权重（全局平均池化梯度）
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 加权求和
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # ReLU
        cam = F.relu(cam)

        # 归一化到 [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    获取目标层（最后一个卷积层）

    Args:
        model: 模型
        model_name: 模型名称

    Returns:
        目标层
    """
    if "efficientnet" in model_name.lower():
        # EfficientNet 的最后一个卷积层
        return model.features[-1]
    else:
        raise ValueError(f"不支持的模型: {model_name}")


def generate_gradcam(
    image_path: Path,
    model_path: Path = None,
    target_class: int = None,
    output_dir: Path = None,
    device: torch.device = None
) -> Dict[str, Path]:
    """
    生成 Grad-CAM 可视化

    Args:
        image_path: 图像路径
        model_path: 模型路径
        target_class: 目标类别（None 表示使用预测类别）
        output_dir: 输出目录
        device: 计算设备

    Returns:
        输出文件路径字典
    """
    # 默认路径
    if model_path is None:
        model_path = config.BEST_MODEL_PATH

    if output_dir is None:
        output_dir = config.GRADCAM_DIR

    if device is None:
        device = get_device()

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # 加载模型
    logging.info(f"加载模型: {model_path}")
    model, _ = load_model_for_inference(str(model_path), model_name=model_name, device=device)

    # 获取目标层
    target_layer = get_target_layer(model, model_name)
    logging.info(f"目标层: {target_layer}")

    # 创建 Grad-CAM
    grad_cam = GradCAM(model, target_layer)

    # 加载图像
    logging.info(f"加载图像: {image_path}")
    original_image = Image.open(image_path).convert('RGB')
    original_image_np = np.array(original_image)

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])

    input_tensor = transform(original_image).unsqueeze(0).to(device)

    # 生成 CAM
    logging.info("生成 Grad-CAM...")
    cam = grad_cam.generate_cam(input_tensor, target_class)

    # 调整 CAM 大小到原始图像大小
    cam_resized = cv2.resize(cam, (original_image_np.shape[1], original_image_np.shape[0]))

    # 生成热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 叠加热力图
    overlay = (heatmap * 0.4 + original_image_np * 0.6).astype(np.uint8)

    # 保存结果
    image_stem = image_path.stem
    original_output_path = output_dir / f"{image_stem}_original.jpg"
    heatmap_output_path = output_dir / f"{image_stem}_heatmap.jpg"
    overlay_output_path = output_dir / f"{image_stem}_overlay.jpg"

    Image.fromarray(original_image_np).save(original_output_path)
    Image.fromarray(heatmap).save(heatmap_output_path)
    Image.fromarray(overlay).save(overlay_output_path)

    logging.info(f"原始图像已保存: {original_output_path}")
    logging.info(f"热力图已保存: {heatmap_output_path}")
    logging.info(f"叠加图已保存: {overlay_output_path}")

    return {
        "original": original_output_path,
        "heatmap": heatmap_output_path,
        "overlay": overlay_output_path
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成 Grad-CAM 可视化")
    parser.add_argument("--image", type=str, required=True, help="图像路径")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--target_class", type=int, default=None, help="目标类别（0=real, 1=fake）")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")

    args = parser.parse_args()

    # 设置日志
    setup_logger()

    # 转换路径
    image_path = Path(args.image)
    model_path = Path(args.model_path) if args.model_path else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    # 生成 Grad-CAM
    output_paths = generate_gradcam(
        image_path,
        model_path,
        args.target_class,
        output_dir
    )

    logging.info("Grad-CAM 生成完成！")


if __name__ == "__main__":
    main()
