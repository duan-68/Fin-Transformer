"""
训练模块
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os

import config
from .data import DualStreamDataset
from .model import FinTransformerModel

def attention_regularization_loss(attn_weights, lambda_entropy=0.01, lambda_sparse=0.001):
    """注意力正则化损失：引导注意力分布符合金融直觉，平衡集中与分散。"""
    entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean()
    sparsity = attn_weights.abs().mean()
    reg_loss = -lambda_entropy * entropy + lambda_sparse * sparsity
    return reg_loss

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_mse = 0
    for X_num, X_text, y, _ in loader:
        X_num, X_text, y = X_num.to(device), X_text.to(device), y.to(device)
        optimizer.zero_grad()
        pred, attn_weights = model(X_num, X_text)
        mse_loss = criterion(pred.squeeze(), y)
        attn_reg = attention_regularization_loss(attn_weights)
        loss = mse_loss + attn_reg  # 总损失用于反向传播（含正则化）
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_mse += mse_loss.item()  # 只记录 MSE 部分，始终非负
    return total_mse / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_num, X_text, y, _ in loader:
            X_num, X_text, y = X_num.to(device), X_text.to(device), y.to(device)
            pred, _ = model(X_num, X_text)
            loss = criterion(pred.squeeze(), y)
            total_loss += loss.item()
    return total_loss / len(loader)

def train_model(model, train_loader, val_loader, epochs, lr, device,
                save_path=None, verbose=True):
    """
    完整的训练流程，返回训练历史。
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
    )

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save(model.state_dict(), save_path)
                if verbose:
                    print(f"  -> 保存最佳模型 (Val Loss: {val_loss:.6f})")

    return history