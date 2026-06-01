"""
预测与反归一化模块
"""

import torch
import numpy as np

def predict_stock(model, X_num, X_text, device, scaler_info):
    """对单个/批量样本进行预测并反归一化。"""
    model.eval()
    X_num_t = torch.FloatTensor(X_num).to(device)
    X_text_t = torch.FloatTensor(X_text).to(device)

    with torch.no_grad():
        pred_norm, attn = model(X_num_t, X_text_t)
        pred_norm = pred_norm.squeeze().cpu().numpy()

    pred_price = pred_norm * scaler_info['close_range'] + scaler_info['close_min']
    return pred_price, attn.cpu().numpy()