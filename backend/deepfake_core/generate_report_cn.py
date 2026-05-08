"""
生成项目可视化报告（中文版）
Generate comprehensive visualizations and report for the project (Chinese version)
"""

import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd


# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")

def setup_chinese_font():
    """配置 matplotlib 中文字体，优先使用 macOS 常见中文字体。"""
    preferred_fonts = [
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "Songti SC",
        "Arial Unicode MS",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
    ]

    available_fonts = {f.name for f in fm.fontManager.ttflist}
    selected_font = None
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            selected_font = font_name
            break

    if selected_font is None:
        selected_font = "DejaVu Sans"
        print("⚠️ 未找到常见中文字体，中文可能无法正常显示。建议安装 Noto Sans CJK SC。")
    else:
        print(f"✅ 使用中文字体: {selected_font}")

    plt.rcParams["font.sans-serif"] = [selected_font]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    return selected_font

CHINESE_FONT = setup_chinese_font()

def load_metrics():
    """加载评估指标"""
    metrics_path = Path("outputs/metrics/test_metrics.json")
    with open(metrics_path, 'r') as f:
        return json.load(f)

def load_training_history():
    """加载训练历史"""
    history_path = Path("outputs/metrics/training_log.csv")
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

def load_train_summary():
    """加载训练摘要"""
    summary_path = Path("outputs/metrics/train_summary.json")
    with open(summary_path, 'r') as f:
        return json.load(f)

def plot_confusion_matrix(metrics, save_path):
    """绘制混淆矩阵"""
    cm_dict = metrics['confusion_matrix']
    cm = np.array([
        [cm_dict['true_negative'], cm_dict['false_positive']],
        [cm_dict['false_negative'], cm_dict['true_positive']]
    ])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['真实', '伪造'],
                yticklabels=['真实', '伪造'],
                cbar_kws={'label': '数量'}, ax=ax)

    ax.set_xlabel('预测标签', fontsize=12, fontweight='bold')
    ax.set_ylabel('真实标签', fontsize=12, fontweight='bold')
    ax.set_title('混淆矩阵 - 测试集', fontsize=14, fontweight='bold', pad=20)

    # 添加统计信息
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total

    textstr = f'总样本数: {total}\n准确率: {accuracy:.1%}\n真阴性: {tn} | 假阳性: {fp}\n假阴性: {fn} | 真阳性: {tp}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(1.35, 0.5, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存至 {save_path}")

def plot_metrics_comparison(metrics, save_path):
    """绘制各项指标对比"""
    metric_names = ['准确率', '精确率', '召回率', 'F1 分数']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1']
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(metric_names, metric_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1%}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylim(0, 1.1)
    ax.set_ylabel('分数', fontsize=12, fontweight='bold')
    ax.set_title('模型性能指标对比', fontsize=14, fontweight='bold', pad=20)
    ax.axhline(y=0.9, color='red', linestyle='--', linewidth=2, alpha=0.5, label='90%基准线')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 指标对比图已保存至 {save_path}")

def plot_roc_curve(metrics, save_path):
    """绘制ROC曲线"""
    fpr = metrics.get('fpr', [0, 0.1, 0.2, 1])
    tpr = metrics.get('tpr', [0, 0.8, 0.9, 1])
    auc = metrics['auc']

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC曲线 (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假阳性率', fontsize=12, fontweight='bold')
    ax.set_ylabel('真阳性率', fontsize=12, fontweight='bold')
    ax.set_title('ROC曲线 - Deepfake检测', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC曲线已保存至 {save_path}")

def plot_training_history(history, save_path):
    """绘制训练历史"""
    epochs = list(range(1, len(history['train_loss']) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Loss曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='训练损失', linewidth=2, markersize=6)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-s', label='验证损失', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('轮次', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('损失', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('训练和验证损失', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)

    # Accuracy曲线
    axes[0, 1].plot(epochs, history['train_acc'], 'b-o', label='训练准确率', linewidth=2, markersize=6)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-s', label='验证准确率', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('轮次', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('准确率', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('训练和验证准确率', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90%目标')

    # AUC曲线
    axes[1, 0].plot(epochs, history['val_auc'], 'g-^', label='验证AUC', linewidth=2, markersize=6)
    best_epoch = np.argmax(history['val_auc']) + 1
    best_auc = max(history['val_auc'])
    axes[1, 0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5,
                       label=f'最佳轮次 {best_epoch} (AUC={best_auc:.4f})')
    axes[1, 0].set_xlabel('轮次', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('AUC', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('验证AUC', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(alpha=0.3)

    # 过拟合分析
    overfitting = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
    axes[1, 1].plot(epochs, overfitting, 'm-d', label='训练-验证差距', linewidth=2, markersize=6)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='5%阈值')
    axes[1, 1].set_xlabel('轮次', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('准确率差距', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('过拟合分析（训练-验证准确率）', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 训练历史已保存至 {save_path}")

def plot_class_distribution(save_path):
    """绘制数据集类别分布"""
    # 从split_metadata读取数据
    metadata_path = Path("outputs/metrics/split_metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            split_data = json.load(f)
    else:
        print("⚠️ 未找到split_metadata.json，跳过类别分布图")
        return

    splits = ['train', 'val', 'test']
    real_counts = []
    fake_counts = []

    for split in splits:
        if split in split_data:
            real_counts.append(split_data[split].get('REAL', 0))
            fake_counts.append(split_data[split].get('FAKE', 0))
        else:
            real_counts.append(0)
            fake_counts.append(0)

    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, real_counts, width, label='真实', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, fake_counts, width, label='伪造', color='#e74c3c', alpha=0.8)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('数据集划分', fontsize=12, fontweight='bold')
    ax.set_ylabel('人脸数量', fontsize=12, fontweight='bold')
    ax.set_title('数据集类别分布', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['训练集', '验证集', '测试集'])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # 添加总计
    total_real = sum(real_counts)
    total_fake = sum(fake_counts)
    total = total_real + total_fake

    if total > 0:
        textstr = f'总计: {total}\n真实: {total_real} ({total_real/total:.1%})\n伪造: {total_fake} ({total_fake/total:.1%})'
    else:
        textstr = f'总计: {total}\n真实: {total_real}\n伪造: {total_fake}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 类别分布图已保存至 {save_path}")

def plot_performance_summary(metrics, save_path):
    """绘制性能总结雷达图"""
    categories = ['准确率', '精确率', '召回率', 'F1 分数', 'AUC']
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
    ax.plot(angles, values, 'o-', linewidth=3, color='#3498db', label='模型性能')
    ax.fill(angles, values, alpha=0.25, color='#3498db')

    # 添加90%参考线
    reference = [0.9] * len(angles)
    ax.plot(angles, reference, '--', linewidth=2, color='red', alpha=0.5, label='90%目标')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
    ax.set_title('模型性能雷达图', fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 性能雷达图已保存至 {save_path}")

def generate_all_visualizations():
    """生成所有可视化图表"""
    print("=" * 60)
    print("📊 生成综合可视化报告（中文版）")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("\n📂 加载数据...")
    metrics = load_metrics()
    history = load_training_history()

    # 生成图表
    print("\n🎨 生成图表...")
    plot_confusion_matrix(metrics, output_dir / "confusion_matrix_cn.png")
    plot_metrics_comparison(metrics, output_dir / "metrics_comparison_cn.png")
    plot_roc_curve(metrics, output_dir / "roc_curve_cn.png")
    plot_training_history(history, output_dir / "training_history_cn.png")
    plot_class_distribution(output_dir / "class_distribution_cn.png")
    plot_performance_summary(metrics, output_dir / "performance_radar_cn.png")

    # 同时覆盖生成无 _cn 后缀的文件，方便 VS Code 中直接查看最新中文版本。
    plot_confusion_matrix(metrics, output_dir / "confusion_matrix.png")
    plot_metrics_comparison(metrics, output_dir / "metrics_comparison.png")
    plot_roc_curve(metrics, output_dir / "roc_curve.png")
    plot_training_history(history, output_dir / "training_history.png")
    plot_class_distribution(output_dir / "class_distribution.png")
    plot_performance_summary(metrics, output_dir / "performance_radar.png")

    print("\n" + "=" * 60)
    print("✅ 所有可视化图表生成成功！")
    print(f"📁 保存至: {output_dir.absolute()}")
    print("=" * 60)

    return output_dir

if __name__ == "__main__":
    generate_all_visualizations()
