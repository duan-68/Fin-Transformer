"""
通用工具函数
"""

import random
import numpy as np
import torch
import platform
import matplotlib.pyplot as plt

def setup_chinese_font():
    """配置 matplotlib 中文显示"""
    system_name = platform.system()
    font_map = {
        'Windows': 'Microsoft YaHei',
        'Darwin': 'Heiti SC',
        'Linux': 'WenQuanYi Micro Hei'
    }
    font_name = font_map.get(system_name, 'DejaVu Sans')
    plt.rcParams['font.sans-serif'] = [font_name, 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    return font_name

def set_seed(seed=42):
    """固定随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False