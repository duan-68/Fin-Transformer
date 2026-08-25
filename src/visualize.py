"""
可视化模块：预测图、训练曲线、SHAP 可解释性图表
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from .utils import setup_chinese_font

setup_chinese_font()

def plot_prediction(real_price, pred_price, stock_code, metrics=None, save_path=None):
    """绘制真实价格与预测价格对比图。"""
    plt.figure(figsize=(16, 8))
    plt.plot(real_price, label='实际收盘价', linewidth=2, color='#1f77b4')
    plt.plot(pred_price, label='预测收盘价 (双流融合)', linestyle='--', linewidth=2, color='#d62728')

    title = f"{stock_code} - FinTransformer融合模型测试结果"
    if metrics:
        title += f"\nRMSE: {metrics['RMSE']:.2f}  MAE: {metrics['MAE']:.2f}  MAPE: {metrics['MAPE']:.2f}%"
        if 'annualized_return' in metrics:
            title += f"  年化收益率: {metrics['annualized_return'] * 100:.2f}%"
        if 'sharpe_ratio' in metrics:
            title += f"  夏普比率: {metrics['sharpe_ratio']:.4f}"
    plt.title(title, fontsize=16)
    plt.xlabel("测试集时间步", fontsize=12)
    plt.ylabel("价格 (CNY)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"预测图已保存至 {save_path}")
    plt.show()

def plot_training_history(history, save_path=None):
    """绘制训练过程中的损失曲线。"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='训练损失', color='#1f77b4')
    plt.plot(history['val_loss'], label='验证损失', color='#d62728')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练与验证损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_shap_summary(shap_results, stock_code, save_dir=None):
    """绘制数值特征重要性柱状图 + 文本时间步重要性曲线。"""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    feat_names = shap_results['feature_names_num']
    importance = shap_results['num_feature_importance']
    importance_norm = importance / importance.sum() * 100

    axes[0].barh(feat_names, importance_norm, color='#1f77b4')
    axes[0].set_xlabel('重要性 (%)')
    axes[0].set_title('数值特征全局重要性')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)

    seq_len = len(shap_results['text_temporal_importance_mean'])
    x_ticks = np.arange(seq_len)
    axes[1].plot(x_ticks, shap_results['text_temporal_importance_mean'],
                 marker='o', color='#d62728', linewidth=2)
    axes[1].set_xlabel('回溯天数（0=最近一天）')
    axes[1].set_title('文本流时间步重要性')
    axes[1].grid(alpha=0.3)

    fig.suptitle(f'{stock_code} SHAP 可解释性分析', fontsize=16)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'shap_summary_{stock_code}.png')
        plt.savefig(path, dpi=150)
        print(f"SHAP 总图已保存至 {path}")
    plt.show()

def plot_shap_beeswarm_violin(shap_results, stock_code, save_dir=None):
    """绘制数值特征的 SHAP 蜂群图 + 小提琴图。"""
    setup_chinese_font()

    shap_values_3d = shap_results['shap_values_num']
    shap_values_2d = shap_values_3d.mean(axis=1)
    feature_names = shap_results['feature_names_num']
    n_samples, n_features = shap_values_2d.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    features_2d = shap_values_3d.mean(axis=1)
    vmin, vmax = features_2d.min(), features_2d.max()
    if vmin == vmax:
        vmin, vmax = vmin - 1, vmax + 1

    cmap = plt.get_cmap('RdBu_r')

    for i in range(n_features):
        shaps = shap_values_2d[:, i]
        feats = features_2d[:, i]
        jitter = np.random.uniform(-0.2, 0.2, size=n_samples)
        y_pos = np.full(n_samples, i) + jitter
        colors = (feats - vmin) / (vmax - vmin)
        ax1.scatter(shaps, y_pos, c=colors, cmap=cmap, s=15, alpha=0.6, edgecolors='none', vmin=0, vmax=1)

    ax1.set_yticks(range(n_features))
    ax1.set_yticklabels(feature_names)
    ax1.set_xlabel('SHAP 值')
    ax1.set_title('SHAP 蜂群图')
    ax1.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()

    violin_data = [shap_values_2d[:, i] for i in range(n_features)]
    parts = ax2.violinplot(violin_data, positions=range(n_features), vert=False, showmeans=True, showmedians=True)

    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_alpha(0.6)
    for key in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
        if key in parts:
            parts[key].set_color('#333333')

    ax2.set_xlabel('SHAP 值')
    ax2.set_title('SHAP 小提琴图')
    ax2.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax2.grid(axis='x', alpha=0.3)

    fig.suptitle(f'{stock_code} 数值特征 SHAP 可解释性分析', fontsize=16)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'shap_beeswarm_violin_{stock_code}.png')
        plt.savefig(path, dpi=150)
        print(f"SHAP 蜂群图+小提琴图已保存至 {path}")
    plt.show()