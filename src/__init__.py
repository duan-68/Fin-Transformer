# 使 src 成为 Python 包
"""
src 包初始化文件
"""

from .data import (
    load_price_data,
    load_text_vectors,
    create_dual_sequences,
    DualStreamDataset,
    run_finbert_inference,
    aggregate_daily_text_vectors
)
from .model import FinTransformerModel, PositionalEncoding
from .train import train_epoch, evaluate, train_model
from .evaluate import evaluate_model, explain_with_shap
from .predict import predict_stock
from .visualize import plot_prediction, plot_training_history
from .utils import setup_chinese_font, set_seed

__all__ = [
    'load_price_data',
    'load_text_vectors',
    'create_dual_sequences',
    'DualStreamDataset',
    'run_finbert_inference',
    'aggregate_daily_text_vectors',
    'FinTransformerModel',
    'PositionalEncoding',
    'train_epoch',
    'evaluate',
    'train_model',
    'evaluate_model',
    'explain_with_shap',
    'predict_stock',
    'plot_prediction',
    'plot_training_history',
    'setup_chinese_font',
    'set_seed'
]