"""
模型定义
基于 EfficientNet 的二分类模型
"""

import logging
import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple

from . import config


def get_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = 2,
    pretrained: bool = True
) -> nn.Module:
    """
    创建 EfficientNet 模型

    Args:
        model_name: 模型名称（efficientnet_b0 或 efficientnet_b3）
        num_classes: 分类数量（默认 2：real/fake）
        pretrained: 是否使用预训练权重

    Returns:
        PyTorch 模型
    """
    logging.info(f"创建模型: {model_name}, 预训练: {pretrained}")

    if model_name == "efficientnet_b0":
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features

    elif model_name == "efficientnet_b3":
        if pretrained:
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features

    else:
        raise ValueError(f"不支持的模型: {model_name}. 支持: efficientnet_b0, efficientnet_b3")

    # 替换分类头（增加 dropout 以减少过拟合）
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features, num_classes)
    )

    logging.info(f"模型创建成功，分类头输入特征: {in_features}, 输出类别: {num_classes}")

    return model


def get_model_info(model: nn.Module) -> dict:
    """
    获取模型信息

    Args:
        model: PyTorch 模型

    Returns:
        模型信息字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_type": model.__class__.__name__
    }

    return info


def load_model_for_inference(
    model_path: str,
    model_name: str = "efficientnet_b0",
    device: torch.device = None
) -> Tuple[nn.Module, dict]:
    """
    加载训练好的模型用于推理

    Args:
        model_path: 模型权重路径
        model_name: 模型名称
        device: 计算设备

    Returns:
        (model, config_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = get_model(model_name=model_name, num_classes=2, pretrained=False)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        model_config = checkpoint.get('config', {})
    else:
        model.load_state_dict(checkpoint)
        model_config = {}

    model.to(device)
    model.eval()

    logging.info(f"已加载模型: {model_path}")

    return model, model_config


class DeepfakeClassifier(nn.Module):
    """
    Deepfake 分类器封装类
    提供更高级的接口
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True
    ):
        """
        初始化分类器

        Args:
            model_name: 模型名称
            num_classes: 分类数量
            pretrained: 是否使用预训练权重
        """
        super(DeepfakeClassifier, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.model = get_model(model_name, num_classes, pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, 3, H, W)

        Returns:
            输出张量 (batch_size, num_classes)
        """
        return self.model(x)

    def get_info(self) -> dict:
        """获取模型信息"""
        return get_model_info(self.model)

    def freeze_backbone(self):
        """冻结骨干网络，只训练分类头"""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

        logging.info("已冻结骨干网络")

    def unfreeze_backbone(self):
        """解冻骨干网络"""
        for param in self.model.parameters():
            param.requires_grad = True

        logging.info("已解冻骨干网络")


def test_model():
    """测试模型创建"""
    print("=" * 60)
    print("测试模型创建")
    print("=" * 60)

    # 测试 EfficientNet-B0
    print("\n1. 创建 EfficientNet-B0")
    model_b0 = get_model("efficientnet_b0", num_classes=2, pretrained=False)
    info_b0 = get_model_info(model_b0)
    print(f"   总参数: {info_b0['total_parameters']:,}")
    print(f"   可训练参数: {info_b0['trainable_parameters']:,}")

    # 测试前向传播
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model_b0(dummy_input)
    print(f"   输入形状: {dummy_input.shape}")
    print(f"   输出形状: {output.shape}")

    # 测试 EfficientNet-B3
    print("\n2. 创建 EfficientNet-B3")
    model_b3 = get_model("efficientnet_b3", num_classes=2, pretrained=False)
    info_b3 = get_model_info(model_b3)
    print(f"   总参数: {info_b3['total_parameters']:,}")
    print(f"   可训练参数: {info_b3['trainable_parameters']:,}")

    # 测试 DeepfakeClassifier
    print("\n3. 创建 DeepfakeClassifier")
    classifier = DeepfakeClassifier("efficientnet_b0", num_classes=2, pretrained=False)
    output = classifier(dummy_input)
    print(f"   输出形状: {output.shape}")

    # 测试冻结/解冻
    print("\n4. 测试冻结/解冻")
    classifier.freeze_backbone()
    trainable_after_freeze = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"   冻结后可训练参数: {trainable_after_freeze:,}")

    classifier.unfreeze_backbone()
    trainable_after_unfreeze = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"   解冻后可训练参数: {trainable_after_unfreeze:,}")

    print("\n" + "=" * 60)
    print("模型测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    test_model()
