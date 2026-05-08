# Deepfake Detection

EfficientNet-based visual Deepfake detection module with frame extraction, face cropping, Grad-CAM explainability, and video-level risk fusion.

基于 EfficientNet 的 Deepfake 视觉检测模块，包含视频抽帧、人脸裁剪、Grad-CAM 可解释性分析和视频级风险融合。

---

## 1. Module Overview / 模块概述

This repository is one part of a larger Deepfake Detection Web App project. It focuses on the visual detection pipeline.

本仓库是 Deepfake 检测 Web App 项目中的一个组成部分，主要负责视觉检测主模型部分。

Main responsibilities:

- Prepare Kaggle DFDC videos using `metadata.json`
- Extract frames from videos
- Detect and crop faces
- Train EfficientNet-B0/B3 binary classifier
- Evaluate model performance
- Generate Grad-CAM visual explanations
- Fuse frame-level predictions into video-level results
- Output visual fake probability and risk score

主要功能：

- 根据 `metadata.json` 整理 Kaggle DFDC 视频
- 视频抽帧
- 人脸检测与裁剪
- 训练 EfficientNet-B0/B3 二分类模型
- 模型评估
- 生成 Grad-CAM 可解释性图像
- 将帧级预测融合为视频级结果
- 输出视觉伪造概率与风险分数

---

## 2. System Architecture / 系统架构

```text
Raw Video
    ↓
Read metadata.json
    ↓
Separate REAL / FAKE videos
    ↓
Frame Extraction
    ↓
Face Detection and Cropping
    ↓
EfficientNet-B0 Classification
    ↓
Frame-level Fake Probability
    ↓
Top Voting / Fusion
    ↓
Video-level Label and Risk Score
    ↓
Grad-CAM Explainability
```

中文流程：

```text
原始视频
    ↓
读取 metadata.json
    ↓
按 REAL / FAKE 标签整理视频
    ↓
视频抽帧
    ↓
人脸检测与裁剪
    ↓
EfficientNet-B0 分类
    ↓
帧级 Fake 概率
    ↓
Top Voting / 融合
    ↓
视频级标签与风险分数
    ↓
Grad-CAM 可解释性分析
```

---

## 3. Dataset / 数据集

### 3.1 Source / 数据来源

Dataset used:

```text
Kaggle Deepfake Detection Challenge Dataset, DFDC
```

The downloaded folder is:

```text
dfdc_train_part_0/
```

It contains:

```text
metadata.json
*.mp4 video files
```

The `metadata.json` file provides the label for each video:

```text
REAL / FAKE
```

Expected placement:

```text
data/kaggle_dfdc/00/
├── metadata.json
├── aaqaifqrwn.mp4
├── aayrffkxzn.mp4
└── ...
```

### 3.2 Final Processed Dataset / 最终处理后数据规模

After video preparation, frame extraction, face detection, and face cropping, the final dataset contains **4706 valid face images**.

经过视频整理、抽帧、人脸检测与裁剪后，最终得到 **4706 张有效人脸图像**。

| Split | Real | Fake | Total |
|---|---:|---:|---:|
| Train | 1070 | 2186 | 3256 |
| Validation | 191 | 480 | 671 |
| Test | 242 | 537 | 779 |
| Total | 1503 | 3203 | 4706 |

Important note:

The split is performed at the **video level**, not randomly at the image level. This prevents frames from the same video appearing in both training and testing sets.

注意：

数据集划分按 **视频级别** 进行，而不是直接随机划分图片。这样可以避免同一个视频的不同帧同时出现在训练集和测试集中，从而减少数据泄漏。

---

## 4. Installation / 安装

### 4.1 Requirements / 环境要求

- Python 3.8+
- PyTorch 2.0+
- torchvision
- OpenCV
- MediaPipe
- scikit-learn
- matplotlib
- pandas
- tqdm
- seaborn

### 4.2 Setup / 安装依赖

```bash
cd deepfake-a-module
pip install -r requirements.txt
```

MediaPipe models may be downloaded automatically during first use.

MediaPipe 模型可能会在首次运行时自动下载。

### 4.3 Dependencies / 依赖包

Main dependencies are listed in `requirements.txt`. The core packages include:

主要依赖包已写入 `requirements.txt`，核心依赖包括：

```text
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
mediapipe>=0.10.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
seaborn
```

---

## 5. Project Structure / 项目结构

```text
deepfake-a-module/
├── data/
│   ├── kaggle_dfdc/00/          # Original Kaggle DFDC chunk
│   ├── raw_videos/              # Videos separated into real/fake
│   ├── frames/                  # Extracted video frames
│   ├── faces/                   # Cropped face images
│   └── dataset/                 # Train/val/test dataset
├── models/
│   ├── best_model.pth           # Best trained EfficientNet model
│   ├── model_config.json        # Model configuration
│   └── mediapipe/               # MediaPipe face model files
├── outputs/
│   ├── gradcam/                 # Grad-CAM visualizations
│   ├── predictions/             # Video prediction JSON files
│   ├── metrics/                 # Metrics, logs, metadata
│   └── visualizations/          # Training and evaluation charts
├── src/
│   ├── config.py                # Global configuration
│   ├── utils.py                 # Utility functions
│   ├── prepare_kaggle_dfdc.py   # Prepare Kaggle DFDC videos
│   ├── extract_frames.py        # Extract frames from videos
│   ├── face_crop.py             # Detect and crop faces
│   ├── build_dataset.py         # Build train/val/test splits
│   ├── model.py                 # EfficientNet model definition
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   ├── predict_image.py         # Single image prediction
│   ├── predict_video.py         # Video-level prediction
│   ├── gradcam.py               # Grad-CAM generation
│   ├── fusion.py                # Video-level fusion logic
│   ├── generate_report.py       # English visualization helper
│   └── generate_report_cn.py    # Chinese visualization helper
├── requirements.txt
└── README.md
```

---

## 6. Full Pipeline / 完整运行流程

### Step 1: Prepare Kaggle DFDC videos / 整理 Kaggle DFDC 视频

```bash
python -m src.prepare_kaggle_dfdc --chunk_dir data/kaggle_dfdc/00 --mode symlink --max_real 100 --max_fake 100
```

This step reads `metadata.json` and separates videos into:

```text
data/raw_videos/real/
data/raw_videos/fake/
```

该步骤读取 `metadata.json`，并按照 REAL / FAKE 标签整理视频。

### Step 2: Extract frames / 视频抽帧

```bash
python -m src.extract_frames
```

Default strategy:

```text
Extract every 10th frame
Maximum 30 frames per video
```

默认策略：

```text
每 10 帧抽取 1 帧
每个视频最多抽取 30 帧
```

Output:

```text
data/frames/real/
data/frames/fake/
outputs/metrics/frame_metadata.csv
```

### Step 3: Detect and crop faces / 人脸检测与裁剪

```bash
python -m src.face_crop
```

This step uses MediaPipe face detection. The largest detected face is cropped, resized to 224×224, and saved.

该步骤使用 MediaPipe 检测人脸，选择最大的人脸区域进行裁剪，并统一调整为 224×224。

Output:

```text
data/faces/real/
data/faces/fake/
outputs/metrics/face_metadata.csv
```

### Step 4: Build dataset / 构建训练集、验证集和测试集

```bash
python -m src.build_dataset
```

Default split:

```text
Train: 70%
Validation: 15%
Test: 15%
```

Output:

```text
data/dataset/train/
data/dataset/val/
data/dataset/test/
outputs/metrics/split_metadata.json
outputs/metrics/split_faces.csv
```

### Step 5: Train model / 训练模型

```bash
python -m src.train
```

Final training configuration:

```text
Model: EfficientNet-B0
Batch size: 16
Learning rate: 5e-5
Weight decay: 1e-5
Image size: 224
Early stopping patience: 5
Best model selection: Validation AUC
```

Output:

```text
models/best_model.pth
models/model_config.json
outputs/metrics/training_log.csv
outputs/metrics/train_summary.json
```

### Step 6: Evaluate model / 测试集评估

```bash
python -m src.evaluate
```

Output:

```text
outputs/metrics/test_metrics.json
```

### Step 7: Predict a single image / 单张图片预测

```bash
python -m src.predict_image --image_path path/to/face.jpg
```

Output includes:

```text
real_probability
fake_probability
predicted_label
```

### Step 8: Predict a video / 视频级预测

```bash
python -m src.predict_video --video_path path/to/video.mp4
```

Output JSON is saved under:

```text
outputs/predictions/
```

Example output fields:

```json
{
  "video_fake_prob": 1.0,
  "video_real_prob": 0.0,
  "video_label": "fake",
  "num_faces": 1,
  "method": "top_vote",
  "risk_score": 100.0,
  "confidence": 1.0,
  "consistency": 1.0,
  "severity": "high",
  "face_stats": {
    "mean_fake_prob": 0.5676,
    "std_fake_prob": 0.0,
    "min_fake_prob": 0.5676,
    "max_fake_prob": 0.5676,
    "median_fake_prob": 0.5676
  }
}
```

---

## 7. Model Details / 模型细节

### 7.1 Backbone / 主干网络

The final model uses:

```text
EfficientNet-B0
```

EfficientNet-B0 is selected because it is lightweight, efficient, and suitable for future Web App deployment.

最终模型使用 **EfficientNet-B0**。选择该模型的原因是它较轻量，推理速度较快，适合后续 Web App 部署。

### 7.2 Model Architecture / 模型结构

```text
EfficientNet-B0 backbone, ImageNet pretrained
    ↓
Global average pooling
    ↓
Dropout
    ↓
Fully connected binary classifier
    ↓
Softmax output: [P(real), P(fake)]
```

类别定义：

```text
real = 0
fake = 1
```

### 7.3 Training Strategy / 训练策略

The final version improves the early overfitting issue by:

- Increasing the dataset size
- Using stronger regularization
- Increasing Dropout
- Lowering the learning rate
- Applying data augmentation
- Using Early Stopping
- Selecting the best model based on validation AUC

最终版本通过以下方式缓解早期过拟合问题：

- 扩大数据量
- 加强正则化
- 提高 Dropout
- 降低学习率
- 使用数据增强
- 使用 Early Stopping
- 按验证集 AUC 保存最佳模型

---

## 8. Final Training Results / 最终训练结果

Final training summary:

| Item | Value |
|---|---:|
| Model | EfficientNet-B0 |
| Total epochs | 17 |
| Best epoch | 12 |
| Best validation AUC | 0.94495 |
| Total training time | 1602.79 seconds, about 26.7 minutes |

Final training metrics:

| Metric | Value |
|---|---:|
| Accuracy | 94.60% |
| Precision | 92.79% |
| Recall | 91.30% |
| F1-score | 92.04% |
| AUC | 98.95% |

Final validation metrics:

| Metric | Value |
|---|---:|
| Accuracy | 86.55% |
| Precision | 76.70% |
| Recall | 85.56% |
| F1-score | 80.89% |
| AUC | 93.65% |

---

## 9. Final Test Results / 最终测试集结果

Final test performance:

| Metric | Value |
|---|---:|
| Accuracy | 91.50% |
| Precision | 90.10% |
| Recall | 84.00% |
| F1-score | 86.94% |
| AUC | 97.42% |

Confusion matrix:

| Type | Count |
|---|---:|
| True Negative | 610 |
| False Positive | 30 |
| False Negative | 52 |
| True Positive | 273 |

Interpretation:

The model achieves strong overall performance. The high AUC of 97.42% indicates that the model has strong ability to distinguish real and fake face images. However, there are still 52 false negative samples, meaning some fake samples are misclassified as real.

结果说明：

模型整体表现较好，AUC 达到 97.42%，说明模型具有较强的真实/伪造区分能力。不过仍存在 52 个假阴性样本，即部分伪造样本被误判为真实。

---

## 10. Video-level Fusion and Risk Scoring / 视频级融合与风险评分

The model predicts each cropped face frame independently. To classify a full video, frame-level predictions are aggregated using a top-voting strategy.

模型首先对每一张裁剪后的人脸帧进行预测。为了得到视频级结果，系统使用 top-vote 策略对帧级预测进行融合。

Output fields:

```text
video_fake_prob
video_real_prob
video_label
num_faces
method
risk_score
confidence
consistency
severity
face_stats
video_path
num_frames
```

Risk levels:

```text
low
medium
high
```

Example video result:

```json
{
  "video_fake_prob": 1.0,
  "video_real_prob": 0.0,
  "video_label": "fake",
  "risk_score": 100.0,
  "confidence": 1.0,
  "severity": "high"
}
```

---

## 11. Grad-CAM Explainability / Grad-CAM 可解释性

Grad-CAM is used to visualize which regions of the face contribute most to the model's fake prediction.

Grad-CAM 用于可视化模型在判断 Deepfake 时主要关注的人脸区域。

Workflow:

```text
Select suspicious frames
    ↓
Forward through EfficientNet
    ↓
Compute class activation map for fake class
    ↓
Generate heatmap
    ↓
Overlay heatmap on original face image
```

Purpose:

- Improve model transparency
- Help explain why a video is considered suspicious
- Provide visualization material for the Web App and final report

作用：

- 提高模型透明度
- 解释模型为什么认为视频可疑
- 为 Web Demo 和最终报告提供可视化材料

---

## 12. Visualization Files / 可视化文件

Useful visualization files:

```text
outputs/visualizations/training_history_cn.png
outputs/visualizations/roc_curve_cn.png
outputs/visualizations/performance_radar_cn.png
outputs/visualizations/metrics_comparison_cn.png
outputs/visualizations/confusion_matrix_cn.png
```

English versions are also available without `_cn` suffix:

```text
outputs/visualizations/training_history.png
outputs/visualizations/roc_curve.png
outputs/visualizations/performance_radar.png
outputs/visualizations/metrics_comparison.png
outputs/visualizations/confusion_matrix.png
```

Auxiliary analysis figures:

```text
outputs/visualizations/fft_sync_availability.png
outputs/visualizations/fft_sync_score_distribution.png
```

`fft_sync_availability.png` shows whether the FFT and Sync submodules can run successfully on the dataset. `fft_sync_score_distribution.png` shows the score distribution and recommended thresholds of the auxiliary modules.

Do not use the old wrong class distribution image if it shows all zeros. Regenerate it from `outputs/metrics/split_metadata.json` or `outputs/metrics/split_faces.csv` before using it in the final report.

如果类别分布图显示全 0，不要放入最终报告。应从 `outputs/metrics/split_metadata.json` 或 `outputs/metrics/split_faces.csv` 重新生成。

---

## 13. Important Notes for Teammates / 给后续组员的重要说明

### 13.1 Do not use old results / 不要使用旧结果

Earlier small-scale experiments used around 50 real and 50 fake videos and produced about 1545 face images. Those results are outdated and should not be used as the final project result.

早期小规模实验使用约 50 个真实视频和 50 个伪造视频，得到约 1545 张人脸图像。该版本结果已经过时，不应作为最终项目结果。

Use these final result files instead:

```text
outputs/metrics/train_summary.json
outputs/metrics/test_metrics.json
outputs/metrics/split_metadata.json
outputs/metrics/training_log.csv
```

最终报告请使用这些文件中的结果。

### 13.2 Files needed by Web App teammate / Web Demo 对接文件

Important files:

```text
models/best_model.pth
models/model_config.json
src/predict_video.py
src/predict_image.py
src/gradcam.py
src/fusion.py
```

The Web App can call `predict_video.py` and display:

```text
video_label
video_fake_prob
risk_score
severity
confidence
consistency
num_faces
face_stats
```

### 13.3 Files needed by auxiliary module teammate / 辅助模块对接文件

The teammate responsible for FFT / Sync / LLM can use A module outputs as visual detection inputs.

Suggested fields to pass forward:

```text
video_fake_prob
risk_score
severity
face_stats
top suspicious frames
Grad-CAM overlay paths
```

## 18. FFT / Sync

该部分负责频域异常检测、音视频同步检测以及 LLM 解释层。其开发流程可以概括为“帧级频域分析 -> 视频级分数归一化 -> 音频与嘴部运动同步分析 -> 与主视觉结果融合 -> LLM 解释输出”，最终为视频级判断提供补充证据。

开发流程如下：

1. 从视频中抽取固定间隔帧，并将 RGB 帧转为灰度图。
2. 分别对每一帧计算 FFT 和 DCT 的高频能量占比。
3. 将每帧结果汇总成视频级 `frequency_score`，并记录高于均值的异常帧索引。
4. 从音频中提取 MFCC 特征，同时从视频中提取嘴部开合序列。
5. 计算音频与嘴部运动之间的时间相关性，并转换为 `sync_score`。
6. 将 `frequency_score`、`sync_score`、主视觉模型输出以及可疑帧证据一起用于最终解释。

FFT/DCT 频域分析的实现如下：

- FFT 部分对每帧灰度图执行二维 FFT，并使用频谱中心与半径阈值 `radius_ratio = 0.35` 将频域划分为低频区和高频区。
- DCT 部分先对灰度图做 0 到 1 归一化，再执行二维 DCT，并以 `keep_low = 0.25` 作为低频块保留比例。
- 两种方法分别得到 `fft_high_freq_ratio` 与 `dct_high_freq_ratio`，再对整段视频求均值作为视频级描述。
- 最终的 `frequency_score` 采用经验式缩放：`clip((raw_score - 0.35) / 0.35, 0, 1)`，其中 `raw_score = 0.5 × fft_mean + 0.5 × dct_mean`。

`frequency_score` 的含义是：越接近 1，表示视频在频域中出现越明显的高频异常，可能与合成残留、纹理断裂或频域伪影有关；越接近 0，则说明当前视频没有达到明显频域异常的水平。当前项目中，`frequency_score` 更适合作为辅助证据，而不是独立最终判定依据。

Sync 音视频同步检测的实现如下：

- 先从视频中提取音频，并使用 `librosa` 计算 MFCC，取第一维系数作为音频时间序列代理。
- 再使用 MediaPipe Face Mesh 追踪嘴部关键点，以上下唇距离近似嘴部开合程度，并用眼距进行尺度归一化。
- 将音频序列与嘴部序列重采样到同一长度后，计算 Pearson 相关系数 `temporal_correlation`。
- 最终同步风险分数定义为 `sync_score = clip(1 - (corr + 1)/2, 0, 1)`，即相关性越高，失同步风险越低。

`sync_score` 的含义是：越接近 0，表示音频与口型运动越同步；越接近 1，表示音画不同步风险越高。对于当前样本而言，`sync_score` 主要用于辅助观察口型与语音是否存在错位，而不是单独决定真假。

该项目已实现一个最小可用的集成：

- `src/fft.py` 返回 `frequency_score`、`fft_high_freq_ratio_mean`、`dct_high_freq_ratio_mean`、标准差以及 `anomaly_frame_indices`。
- `src/sync.py` 返回 `sync_score`、`temporal_correlation`、`num_samples` 和 `fps`。
- `src/llm.py` 接收 `suspicious_frames`、`gradcam_paths`、`frequency_summary`、`sync_summary`，并将这些信息格式化为更详细的自然语言 prompt。

`fft_sync_availability.png` 建议放在 FFT / Sync 模块可运行性分析之后，用来说明两个子模块在数据集上是否能够稳定运行；`fft_sync_score_distribution.png` 建议放在频域阈值和同步阈值解释之后，用来展示两类分数的分布以及推荐阈值。

注意：当前 LLM 解释仍依赖你传入的证据字段（可疑帧索引、Grad-CAM 覆盖图）。如果希望 LLM 精确指出 "画面哪里不自然"，建议确保 Grad-CAM 输出覆盖到可疑帧并把热图路径传入 LLM。

### frequency_score 解释

frequency_score 的计算逻辑（实现参考 `src/fft.py`）：

1. 对每一帧计算两类比率：`fft_high_freq_ratio`（基于 2D FFT 的高频能量占比）和 `dct_high_freq_ratio`（基于 DCT 的高频能量占比）。
2. 分别求两类比率的平均值：`fft_mean` 与 `dct_mean`。
3. 计算原始分数：`raw_score = 0.5 * fft_mean + 0.5 * dct_mean`。
4. 线性缩放并裁剪：`frequency_score = clip((raw_score - 0.35) / 0.35, 0, 1)`。

当前标定结果显示，`frequency_threshold = 0.0`，这说明在当前数据集上，大多数视频的频域分数都较低，因此该指标更适合作为“是否存在异常高频信号”的辅助筛查项，而不是单独的最终判定标准。

含义：`frequency_score` 越接近 1，表示视频在频域中观测到较强的高频异常（可能由合成残留或频域伪影引起）。当 `raw_score` 小于 0.35 时，会被截断为 0（表示未检出明显频域异常）。此外，函数也会返回每帧比率与基于均值+1*std 的异常帧索引，便于 LLM 给出时序证据。

### 关于 LLM API Key 与开源模型

- OpenAI：在 `src/llm.py` 中默认模型名为 `gpt-4o-mini`，要真正调用 OpenAI，需要在运行环境中设置环境变量 `OPENAI_API_KEY`。Key 可在 https://platform.openai.com 创建并管理（需要注册并可能有计费）。云端调用会产生按模型定价计费。\
    在 Windows cmd 中临时设置（仅当前窗口有效）：

```powershell
set OPENAI_API_KEY=sk-...yourkey...
```

或者永久设置：

```powershell
setx OPENAI_API_KEY "sk-...yourkey..."
```

- 开源/本地替代：若不希望付费或希望离线运行，可使用开源模型并在本地用类似 `text-generation-webui` / `llama.cpp` / HuggingFace 运行。优缺点：
    - 本地运行（如基于 GGML 的量化 Llama / Mistral）可以无限制使用但对硬件（尤其显存）要求高；小模型（7B/13B 量化）在普通 GPU/甚至 CPU 上可用但能力有限。\
    - Hugging Face Inference API 或其他服务有免费额度（按需变化），但通常需要注册并获取 token（同样是按提供方的条款）。

- DeepSeek：官方 API 文档显示 DeepSeek 使用 OpenAI 兼容格式，调用时需要申请 `DEEPSEEK_API_KEY`，并按 token 计费。如果要做这个项目里最合适、成本最低的云端方案，建议优先使用 `deepseek-v4-flash`，它是当前文档里面更适合文本解释层的低成本模型。
    在 Windows cmd 中临时设置：

```powershell
set DEEPSEEK_API_KEY=sk-...yourkey...
set DEEPSEEK_BASE_URL=https://api.deepseek.com
set LLM_PROVIDER=deepseek
```

默认模型建议使用：

```text
deepseek-v4-flash
```

如果你未来想切到思考模式，可改为 `deepseek-v4-pro`，但通常成本更高；`deepseek-chat` 和 `deepseek-reasoner` 已在文档中标记为后续会弃用的兼容名。

现实选择往往是：在本地部署一个开源模型（如经过量化的 Llama / Mistral）并通过本地 HTTP 接口供 `generate_llm_explanation` 调用；这需要额外安装和配置（超出本说明范围），但能避免按调用计费。

---

Possible final fusion formula:

```text
Final Risk Score = 0.60 × Visual Risk Score + 0.25 × Frequency Risk Score + 0.15 × Sync Risk Score
```

If Sync is unstable, use:

```text
Final Risk Score = 0.75 × Visual Risk Score + 0.25 × Frequency Risk Score
```

---

## 14. Common Issues / 常见问题

### 14.1 MediaPipe cannot detect faces in some frames / 部分帧无法检测到人脸

This is expected for frames with side faces, motion blur, occlusion, or small faces. The pipeline skips frames without valid face detection and records them in metadata files.

对于侧脸、运动模糊、遮挡或人脸过小的画面，MediaPipe 可能无法检测到人脸。当前流程会跳过这些无效帧，并在 metadata 文件中记录。

### 14.2 Training is slow / 训练速度较慢

Use EfficientNet-B0 instead of EfficientNet-B3, reduce batch size, or reduce the number of sampled frames per video.

如果训练较慢，可以使用 EfficientNet-B0 而不是 EfficientNet-B3，降低 batch size，或减少每个视频抽取的帧数。

### 14.3 Out of memory / 显存或内存不足

Reduce batch size first. If the issue remains, use a smaller image size or reduce the number of training samples.

优先降低 batch size。如果仍然内存不足，可以降低图像尺寸或减少训练样本数量。

### 14.4 Class distribution chart shows all zeros / 类别分布图显示全 0

Do not use the wrong generated chart. Regenerate it from `outputs/metrics/split_metadata.json` or `outputs/metrics/split_faces.csv`.

如果类别分布图显示全 0，不要将该图放入报告。应从 `outputs/metrics/split_metadata.json` 或 `outputs/metrics/split_faces.csv` 重新生成。

---

## 15. Limitations and Future Work / 局限性与未来改进

Current limitations:

- Some fake videos may still be misclassified as real.
- Video-level prediction may be unstable when only a few valid face frames are detected.
- The top-vote fusion strategy can be further improved.
- Grad-CAM provides visual explanation but does not fully explain all model decisions.

当前局限：

- 仍有部分伪造视频可能被误判为真实。
- 当有效人脸帧数量较少时，视频级预测可能不够稳定。
- top-vote 视频级融合策略仍可进一步优化。
- Grad-CAM 能提供可视化解释，但不能完全解释所有模型决策。

Future improvements:

- Add minimum valid-face-frame threshold.
- Add LLM-based natural language explanation.
- Train on more DFDC chunks or additional datasets.
- Improve Web App integration and result visualization.

未来改进方向：

- 增加最少有效人脸帧数量限制。
- 加入 LLM 自然语言解释层。
- 使用更多 DFDC 分块或其他数据集训练。
- 优化 Web App 展示效果。

---

## 16. Acknowledgements / 致谢

- EfficientNet: Tan & Le, 2019
- Grad-CAM: Selvaraju et al., 2017
- MediaPipe: Google Research
- DFDC Dataset: Facebook AI / Kaggle

---

## 17. Summary / 总结

This module completes the main visual detection pipeline for the Deepfake Detection Web App project. It processes Kaggle DFDC videos into face images, trains an EfficientNet-B0 classifier, evaluates the model, generates video-level risk scores, and provides Grad-CAM explainability support.

本模块完成了 Deepfake 检测 Web App 项目中的主视觉检测流程。系统能够将 Kaggle DFDC 视频处理为人脸图像，训练 EfficientNet-B0 分类模型，进行测试集评估，生成视频级风险分数，并提供 Grad-CAM 可解释性支持。

Final test performance:

```text
Accuracy: 91.50%
Precision: 90.10%
Recall: 84.00%
F1-score: 86.94%
AUC: 97.42%
```

## 19. Recent Modifications / 最近变更记录

This section documents code changes, bug fixes, new utilities, and exact commands to reproduce recent inference and calibration experiments.  
本节记录近期对代码的修改、错误修复、新增脚本，以及可复现的运行命令。

- **Files changed / 修改的文件**
    - `src/predict_video.py` :
        - Default fusion method changed from `top_vote` to `weighted_average` (CLI and function signature).  
        - Fixed calibrator usage: build a proper DataFrame of features before calling `predict_proba` so FFT/Sync scores participate in calibrated fusion.  
        - Added/OpenCV Haar-cascade fallback when MediaPipe face detection fails on some frames/videos.  
        - Result: single-video CLI now uses `weighted_average` by default and applies the saved calibrator when available.
    - `src/batch_recompute_compare.py` :
        - New script added to recompute video-level predictions for multiple fusion methods (e.g., `average`, `top_vote`, `weighted_average`) and produce per-video records + summary CSV/JSON.  
        - Fixed metadata lookup bug by normalizing keys to video stems (strip `.mp4`) so `true_label` is retrieved correctly.
    - `src/fusion.py` :
        - (Documentation note) Fusion logic supports `average`, `max`, `top_vote`, `weighted_average` — weighted_average recommended after calibration runs.
    - `src/fft.py`, `src/sync.py` :
        - No API-breaking changes; used by `predict_video` and batch runs. Ensure `--enable_fft` / `--enable_sync` flags are passed to enable these modules.
    - `src/llm.py` :
        - No change needed for current runs; LLM explanation consumes suspicious frames / gradcam paths when provided.
    - `src/batch_recompute_compare.py` outputs written to `outputs/metrics/` (see paths below).
    - `models/fusion_calibrator.pkl` :
        - New post-hoc calibrator file (XGBoost selected from logistic/rf/xgb experiments). This is used by `predict_video` when present.
    - `src/config.py` :
        - `THRESHOLD` scanned and set to `0.49` based on calibrated threshold sweep.

- **Bug fixes / 关键修复**
    - Metadata key mismatch: batch script previously looked up metadata keys that included `.mp4` while the code used `Path.stem`; fixed by normalizing metadata keys to stems. This removed `null` `true_label` entries in batch outputs.
    - Calibrator input shape: fixed passing of features to calibrator so `frequency_score` and `sync_score` are included correctly in calibrated probabilities.
    - Face detection fallback: when MediaPipe fails to return faces for a frame/video, fall back to OpenCV Haar cascade to increase coverage.

- **New scripts & utilities / 新增脚本**
    - `src/batch_recompute_compare.py` — recompute predictions across fusion methods and produce `outputs/metrics/fusion_method_comparison.json` and `.csv`.
    - Calibration training & scanning utilities (used earlier to produce `models/fusion_calibrator.pkl` and scan `THRESHOLD`) exist under `src/` (see `src/` files related to calibration).

- **Reproducible commands / 可复现命令**
    - Single-video inference (uses default `weighted_average` if no `--method` given):

```bash
python -m src.predict_video --video_path data/test/test.mp4 --enable_fft --enable_sync
```

    - Batch recompute & compare (example run used during analysis):

```bash
python -m src.batch_recompute_compare \
    --video_dir data/kaggle_dfdc/00 \
    --metadata data/kaggle_dfdc/00/metadata.json \
    --out_dir outputs/metrics \
    --limit 50 \
    --enable_fft \
    --enable_sync
```

    - If you run under a specialized conda env (recommended to avoid binary mismatches), replace `python` with your env executable, e.g.:

```powershell
D:\Anaconda\envs\deepfakebench\python.exe -m src.batch_recompute_compare --video_dir ...
```

- **Outputs produced / 产出文件**
    - `outputs/metrics/fusion_method_comparison.json` — summary and per-video records for each fusion method (one record per video per method).  
    - `outputs/metrics/fusion_method_comparison.csv` — flattened CSV of the same records.  
    - `models/fusion_calibrator.pkl` — saved calibrator (XGBoost) used in calibrated fusion.  
    - Example fields included per record: `video_id` (stem), `video_path`, `method`, `predicted_label`, `predicted_prob`, `true_label`, `correct`, `num_faces`.

- **Recommended default & rationale / 推荐默认与理由**
    - Recommended fusion: **`weighted_average`** (now set as default in `src/predict_video.py`).  
    - Rationale: In our 50-video sample runs, `average` and `weighted_average` tied for best accuracy (~0.94), while `top_vote` performed worse (~0.74). Weighted average incorporates frame-level confidence and auxiliary FFT/Sync evidence after calibration, providing robust video-level scores.

- **Where to look for traces of changes / 在哪里查看修改痕迹**
    - See the edited files in `src/` for exact code edits: `src/predict_video.py`, `src/batch_recompute_compare.py`, and `src/fusion.py` for fusion logic.  
    - Generated outputs (examples used in evaluation) are under `outputs/metrics/`.

