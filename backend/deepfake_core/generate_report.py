"""
生成项目可视化报告
Generate comprehensive visualizations and report for the project
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# English report: use default matplotlib font to avoid Chinese font issues.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_metrics():
    """加载评估指标"""
    metrics_path = PROJECT_ROOT / "outputs" / "metrics" / "test_metrics.json"
    with open(metrics_path, 'r') as f:
        return json.load(f)

def load_training_history():
    """加载训练历史"""
    history_path = PROJECT_ROOT / "outputs" / "metrics" / "training_log.csv"
    df = pd.read_csv(history_path)

    return {
        'train_loss': df['train_loss'].tolist(),
        'train_acc': df['train_acc'].tolist(),
        'train_f1': df['train_f1'].tolist(),
        'train_auc': df['train_auc'].tolist(),
        'val_loss': df['val_loss'].tolist(),
        'val_acc': df['val_acc'].tolist(),
        'val_f1': df['val_f1'].tolist(),
        'val_auc': df['val_auc'].tolist(),
    }

def load_module_b_outputs():
    """加载频域/同步标定结果与逐视频分数（若存在）"""
    calibration_path = PROJECT_ROOT / "outputs" / "metrics" / "fft_sync_calibration.json"
    scores_path = PROJECT_ROOT / "outputs" / "metrics" / "fft_sync_scores.csv"

    calibration = None
    scores_df = None

    if calibration_path.exists():
        with open(calibration_path, 'r', encoding='utf-8') as f:
            calibration = json.load(f)

    if scores_path.exists():
        scores_df = pd.read_csv(scores_path)

        # 标准化标签与可用性字段，避免绘图时类型不一致。
        if 'label' in scores_df.columns:
            scores_df['label'] = scores_df['label'].astype(str).str.lower()

        for c in ['freq_available', 'sync_available']:
            if c in scores_df.columns:
                scores_df[c] = scores_df[c].astype(str).str.lower().isin(['true', '1', 'yes'])

        for c in ['frequency_score', 'sync_score']:
            if c in scores_df.columns:
                scores_df[c] = pd.to_numeric(scores_df[c], errors='coerce')

    return calibration, scores_df

def plot_confusion_matrix(metrics, save_path):
    """绘制混淆矩阵"""
    cm_dict = metrics['confusion_matrix']
    cm = np.array([
        [cm_dict['true_negative'], cm_dict['false_positive']],
        [cm_dict['false_negative'], cm_dict['true_positive']]
    ])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Count'}, ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold', pad=20)

    # 添加统计信息
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total

    textstr = f'Total samples: {total}\nAccuracy: {accuracy:.1%}\nTN: {tn} | FP: {fp}\nFN: {fn} | TP: {tp}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(1.35, 0.5, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to {save_path}")

def plot_metrics_comparison(metrics, save_path):
    """绘制各项指标对比"""
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1']
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(metric_names, metric_values,
                  alpha=0.8, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1%}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.axhline(y=0.9, linestyle='--', linewidth=2, alpha=0.5, label='90% Target')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Metrics comparison chart saved to {save_path}")

def plot_roc_curve(metrics, save_path):
    """绘制ROC曲线"""
    fpr = metrics.get('fpr', [0, 0.1, 0.2, 1])
    tpr = metrics.get('tpr', [0, 0.8, 0.9, 1])
    auc = metrics['auc']

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, lw=3, label=f'ROC Curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve - Deepfake Detection', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curve saved to {save_path}")

def plot_training_history(history, save_path):
    """绘制训练历史"""
    epochs = list(range(1, len(history['train_loss']) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Loss曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)

    # Accuracy曲线
    axes[0, 1].plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% Target')

    # AUC曲线
    axes[1, 0].plot(epochs, history['val_auc'], 'g-^', label='Validation AUC', linewidth=2, markersize=6)
    best_epoch = np.argmax(history['val_auc']) + 1
    best_auc = max(history['val_auc'])
    axes[1, 0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5,
                       label=f'Best Epoch {best_epoch} (AUC={best_auc:.4f})')
    axes[1, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('AUC', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Validation AUC', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(alpha=0.3)

    # 过拟合分析
    overfitting = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
    axes[1, 1].plot(epochs, overfitting, 'm-d', label='Train-Val Gap', linewidth=2, markersize=6)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='5% Threshold')
    axes[1, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Accuracy Gap', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Overfitting Analysis (Train - Validation Accuracy)', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Training history saved to {save_path}")

def plot_class_distribution(save_path):
    """绘制数据集类别分布"""
    splits = ['train', 'val', 'test']
    split_names = ['Train', 'Validation', 'Test']
    real_counts = []
    fake_counts = []

    split_faces_path = Path("outputs/metrics/split_faces.csv")
    split_metadata_path = Path("outputs/metrics/split_metadata.json")

    if split_faces_path.exists():
        df = pd.read_csv(split_faces_path)
        split_col = 'split' if 'split' in df.columns else None
        label_col = 'label' if 'label' in df.columns else None

        if split_col and label_col:
            for split in splits:
                split_df = df[df[split_col].astype(str).str.lower() == split]
                labels = split_df[label_col].astype(str).str.lower()
                real_counts.append(int((labels == 'real').sum()))
                fake_counts.append(int((labels == 'fake').sum()))
        else:
            print("⚠️ split_faces.csv does not contain expected split/label columns.")
            real_counts = [0, 0, 0]
            fake_counts = [0, 0, 0]
    elif split_metadata_path.exists():
        with open(split_metadata_path, 'r') as f:
            metadata = json.load(f)
        for split in splits:
            split_data = metadata.get(split, {})
            real_counts.append(int(split_data.get('real', 0)))
            fake_counts.append(int(split_data.get('fake', 0)))
    else:
        print("⚠️ No split metadata found. Class distribution will show zeros.")
        real_counts = [0, 0, 0]
        fake_counts = [0, 0, 0]

    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, real_counts, width, label='Real', alpha=0.8)
    bars2 = ax.bar(x + width/2, fake_counts, width, label='Fake', alpha=0.8)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Dataset Split', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Face Images', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Class Distribution', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(split_names)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # 添加总计
    total_real = sum(real_counts)
    total_fake = sum(fake_counts)
    total = total_real + total_fake

    if total > 0:
        textstr = f'Total: {total}\nReal: {total_real} ({total_real/total:.1%})\nFake: {total_fake} ({total_fake/total:.1%})'
    else:
        textstr = f'Total: {total}\nReal: {total_real}\nFake: {total_fake}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Class distribution chart saved to {save_path}")

def plot_performance_summary(metrics, save_path):
    """绘制性能总结雷达图"""
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['auc']
    ]

    # 闭合雷达图
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=3, label='Model Performance')
    ax.fill(angles, values, alpha=0.25)

    # 添加90%参考线
    reference = [0.9] * len(angles)
    ax.plot(angles, reference, '--', linewidth=2, alpha=0.5, label='90% Target')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
    ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Performance radar chart saved to {save_path}")

def plot_module_b_score_distribution(scores_df, calibration, save_path):
    """绘制频域/同步分数分布（含推荐阈值）"""
    if scores_df is None or scores_df.empty:
        print("⚠️ Score table is empty. Skip score distribution plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    labels = ['real', 'fake']
    palette = {'real': '#1f77b4', 'fake': '#d62728'}

    if 'frequency_score' in scores_df.columns:
        freq_df = scores_df.dropna(subset=['frequency_score']).copy()
        if 'label' in freq_df.columns and not freq_df.empty:
            sns.histplot(
                data=freq_df[freq_df['label'].isin(labels)],
                x='frequency_score',
                hue='label',
                bins=30,
                stat='density',
                common_norm=False,
                alpha=0.35,
                palette=palette,
                ax=axes[0],
            )
        else:
            axes[0].hist(freq_df['frequency_score'].dropna(), bins=30, alpha=0.7, color='#1f77b4')

        freq_th = None
        if calibration:
            freq_th = calibration.get('recommended', {}).get('frequency_threshold')
        if freq_th is not None:
            axes[0].axvline(freq_th, color='black', linestyle='--', linewidth=2, label=f"threshold={freq_th:.4f}")
            axes[0].legend()

    axes[0].set_title('Frequency Score Distribution')
    axes[0].set_xlabel('frequency_score')
    axes[0].set_ylabel('Density')

    if 'sync_score' in scores_df.columns:
        sync_df = scores_df.dropna(subset=['sync_score']).copy()
        if 'label' in sync_df.columns and not sync_df.empty:
            sns.histplot(
                data=sync_df[sync_df['label'].isin(labels)],
                x='sync_score',
                hue='label',
                bins=30,
                stat='density',
                common_norm=False,
                alpha=0.35,
                palette=palette,
                ax=axes[1],
            )
        else:
            axes[1].hist(sync_df['sync_score'].dropna(), bins=30, alpha=0.7, color='#d62728')

        sync_th = None
        if calibration:
            sync_th = calibration.get('recommended', {}).get('sync_threshold')
        if sync_th is not None:
            axes[1].axvline(sync_th, color='black', linestyle='--', linewidth=2, label=f"threshold={sync_th:.4f}")
            axes[1].legend()

    axes[1].set_title('Sync Score Distribution')
    axes[1].set_xlabel('sync_score')
    axes[1].set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Score distribution saved to {save_path}")

def plot_module_b_availability(scores_df, save_path):
    """绘制两个子功能可用率"""
    if scores_df is None or scores_df.empty:
        print("⚠️ Score table is empty. Skip availability plot.")
        return

    freq_ok = float(scores_df['freq_available'].mean()) if 'freq_available' in scores_df.columns else np.nan
    sync_ok = float(scores_df['sync_available'].mean()) if 'sync_available' in scores_df.columns else np.nan

    labels = ['FFT available', 'Sync available']
    values = [freq_ok, sync_ok]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=['#2ca02c', '#ff7f0e'], alpha=0.85, edgecolor='black')

    for bar, val in zip(bars, values):
        if np.isnan(val):
            txt = 'N/A'
            y = 0.02
        else:
            txt = f"{val:.1%}"
            y = val + 0.02
        ax.text(bar.get_x() + bar.get_width() / 2, y, txt, ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Availability Rate')
    ax.set_title('Submodule Availability')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Availability chart saved to {save_path}")

def generate_all_visualizations():
    """生成所有可视化图表"""
    print("=" * 60)
    print("📊 Generating Comprehensive Visualizations")
    print("=" * 60)

    # 创建输出目录
    output_dir = PROJECT_ROOT / "outputs" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Delete old report images before regenerating them.
    english_files = [
        "confusion_matrix.png",
        "metrics_comparison.png",
        "roc_curve.png",
        "training_history.png",
        "class_distribution.png",
        "performance_radar.png",
        "fft_sync_score_distribution.png",
        "fft_sync_availability.png",
    ]
    for filename in english_files:
        old_file = output_dir / filename
        if old_file.exists():
            old_file.unlink()
            print(f"🗑️ Deleted old file: {old_file}")

    # 加载数据
    print("\n📂 Loading data...")
    metrics = load_metrics()
    history = load_training_history()
    module_b_calibration, module_b_scores = load_module_b_outputs()

    # 生成图表
    print("\n🎨 Generating charts...")
    plot_confusion_matrix(metrics, output_dir / "confusion_matrix.png")
    plot_metrics_comparison(metrics, output_dir / "metrics_comparison.png")
    plot_roc_curve(metrics, output_dir / "roc_curve.png")
    plot_training_history(history, output_dir / "training_history.png")
    plot_class_distribution(output_dir / "class_distribution.png")
    plot_performance_summary(metrics, output_dir / "performance_radar.png")

    # Auxiliary analysis charts: only generate when calibration / per-video results exist.
    if module_b_scores is not None:
        plot_module_b_score_distribution(module_b_scores, module_b_calibration, output_dir / "fft_sync_score_distribution.png")
        plot_module_b_availability(module_b_scores, output_dir / "fft_sync_availability.png")
    else:
        print("⚠️ Module B outputs not found. Skipped Module B plots.")

    print("\n" + "=" * 60)
    print("✅ All visualizations generated successfully!")
    print(f"📁 Saved to: {output_dir.absolute()}")
    print("=" * 60)

    return output_dir

if __name__ == "__main__":
    generate_all_visualizations()
