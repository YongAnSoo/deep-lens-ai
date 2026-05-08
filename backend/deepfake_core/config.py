"""
Deepfake 检测模块 A 的配置文件
存储所有路径、超参数和设置
"""

from pathlib import Path
import torch

# ============================================================================
# 项目路径
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
KAGGLE_DFDC_DIR = DATA_DIR / "kaggle_dfdc" / "00"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
FRAMES_DIR = DATA_DIR / "frames"
FACES_DIR = DATA_DIR / "faces"
DATASET_DIR = DATA_DIR / "dataset"

# 模型目录
MODELS_DIR = PROJECT_ROOT / "models"
MEDIAPIPE_MODEL_PATH = MODELS_DIR / "mediapipe" / "blaze_face_short_range.tflite"

# 输出目录
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
METRICS_DIR = OUTPUTS_DIR / "metrics"

# ============================================================================
# 数据处理参数
# ============================================================================
# 帧提取
FRAME_INTERVAL = 10  # 每隔 N 帧提取一帧
MAX_FRAMES_PER_VIDEO = 30  # 每个视频最多提取的帧数

# 人脸检测和裁剪
FACE_MARGIN = 0.25  # 检测到的人脸周围的边距（25%）
IMAGE_SIZE = 224  # 模型输入图像大小（224x224）

# 数据集划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 随机种子，用于可重复性
RANDOM_SEED = 42

# ============================================================================
# 模型参数
# ============================================================================
MODEL_NAME = "efficientnet_b0"  # 选项: efficientnet_b0, efficientnet_b3
NUM_CLASSES = 2  # 二分类: real (0) vs fake (1)
PRETRAINED = True

# 类别映射
# ImageFolder 会按字母序为 data/train/fake 和 data/train/real 分配索引：fake=0, real=1
CLASS_NAMES = ["fake", "real"]
FAKE_CLASS = 0
REAL_CLASS = 1

# ============================================================================
# 训练参数
# ============================================================================
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 5e-5  # 降低学习率以提高稳定性
WEIGHT_DECAY = 1e-5

# 早停
PATIENCE = 5  # 如果 N 个 epoch 没有改进则停止

# 模型检查点
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

# ============================================================================
# 预测参数
# ============================================================================
THRESHOLD = 0.49  # 假视频检测分类阈值（基于校准后视频级准确率扫描）
TOP_K = 5  # Grad-CAM 的最可疑帧数量
TOP_VOTE_PERCENT = 0.3  # 使用前 30% 的帧进行视频级投票

# 风险等级阈值
RISK_LOW_THRESHOLD = 40
RISK_MEDIUM_THRESHOLD = 70

# ============================================================================
# 设备配置
# ============================================================================
def get_device():
    """
    确定可用的最佳计算设备
    优先级: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# ============================================================================
# ImageNet 归一化（用于预训练模型）
# ============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# 日志记录
# ============================================================================
LOG_INTERVAL = 10  # 训练期间每 N 个批次记录一次
