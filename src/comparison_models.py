"""
对比模型定义：LSTM、BiLSTM-Attention、Informer
用于与 Fin-Transformer 进行基线对比实验。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from src.model import PositionalEncoding


class LSTMModel(nn.Module):
    """纯数值 LSTM 基线模型

    使用两层 LSTM 编码数值时序特征，取最后时间步隐状态经 MLP 预测。
    """

    def __init__(self, num_input_dim=7, d_model=128, num_layers=2,
                 dropout=0.1, pred_length=1, dim_feedforward=256, **kwargs):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_input_dim,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.pred_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, pred_length),
        )

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x_num)
        last_step = lstm_out[:, -1, :]
        pred = self.pred_head(last_step)
        return pred


class BiLSTMAttentionModel(nn.Module):
    """双流 BiLSTM + 时间注意力融合模型

    数值流与文本流分别使用 BiLSTM 编码，拼接后通过时间维度注意力聚合上下文。
    """

    def __init__(self, num_input_dim=7, text_input_dim=768, d_model=128,
                 num_layers=2, dropout=0.1, pred_length=1, **kwargs):
        super().__init__()

        hidden_size = d_model // 2

        self.num_bilstm = nn.LSTM(
            input_size=num_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        self.text_proj = nn.Linear(text_input_dim, d_model)
        self.text_bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        fused_dim = d_model * 2
        self.attn_score = nn.Linear(fused_dim, 1)

        self.pred_head = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_length),
        )

    def forward(self, x_num: torch.Tensor, x_text: torch.Tensor) -> Tuple[torch.Tensor, None]:
        num_out, _ = self.num_bilstm(x_num)

        text_feat = self.text_proj(x_text)
        text_out, _ = self.text_bilstm(text_feat)

        fused = torch.cat([num_out, text_out], dim=-1)

        score = self.attn_score(fused).squeeze(-1)
        weights = F.softmax(score, dim=-1)
        context = (weights.unsqueeze(-1) * fused).sum(dim=1)

        pred = self.pred_head(context)
        return pred, None


class ProbSparseAttention(nn.Module):
    """ProbSparse 自注意力机制 (Informer, AAAI 2021)

    从所有 query 中选出 top-u 个活跃 query 进行完整注意力计算，
    其余 query 使用 V 的均值填充，将复杂度从 O(L^2) 降到 O(L·log L)。
    """

    def __init__(self, d_model=128, nhead=4, factor=5, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.factor = factor

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _prob_QK(self, Q: torch.Tensor, K: torch.Tensor,
                 sample_k: int, n_top: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, L_K, D = K.shape
        _, _, L_Q, _ = Q.shape

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        idx_sample = torch.randint(L_K, (L_Q, sample_k), device=K.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), idx_sample, :]

        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        M = Q_K_sample.max(-1)[0] - Q_K_sample.mean(-1)

        M_top = M.topk(n_top, sorted=False)[1]
        Q_top = Q.gather(2, M_top.unsqueeze(-1).expand(-1, -1, -1, D))

        return Q_top, M_top

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H = self.nhead
        D = self.d_k

        Q = self.W_Q(x).view(B, L, H, D).transpose(1, 2)
        K = self.W_K(x).view(B, L, H, D).transpose(1, 2)
        V = self.W_V(x).view(B, L, H, D).transpose(1, 2)

        U = max(1, int(self.factor * math.ceil(math.log(L + 1))))
        u = min(U, L)
        sample_k = min(max(1, int(self.factor * math.ceil(math.log(L + 1)))), L)

        V_mean = V.mean(dim=2, keepdim=True).expand(-1, -1, L, -1)
        context = V_mean.clone()

        Q_top, top_idx = self._prob_QK(Q, K, sample_k, u)

        scale = 1.0 / math.sqrt(D)
        attn_scores = torch.matmul(Q_top, K.transpose(-2, -1)) * scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_out = torch.matmul(attn_probs, V)

        context.scatter_(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, D), attn_out)

        context = context.transpose(1, 2).contiguous().view(B, L, H * D)
        out = self.out_proj(context)
        return out


class InformerEncoderLayer(nn.Module):
    """Informer 编码器单层：ProbSparse 注意力 + FFN + 残差 + LayerNorm"""

    def __init__(self, d_model=128, nhead=4, dim_feedforward=256, factor=5, dropout=0.1):
        super().__init__()

        self.self_attn = ProbSparseAttention(d_model, nhead, factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class InformerModel(nn.Module):
    """Informer 编码器模型 (AAAI 2021)

    使用 ProbSparse 自注意力机制替代标准 Transformer 的全注意力，
    在长序列预测场景下具有更高效的计算复杂度。
    """

    def __init__(self, num_input_dim=7, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1,
                 pred_length=1, factor=5, **kwargs):
        super().__init__()

        self.input_proj = nn.Linear(num_input_dim, d_model)
        self.scale = math.sqrt(d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, nhead, dim_feedforward, factor, dropout)
            for _ in range(num_layers)
        ])

        self.pred_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, pred_length),
        )

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        feat = self.input_proj(x_num) * self.scale
        feat = self.pos_encoder(feat)

        for layer in self.encoder_layers:
            feat = layer(feat)

        last_step = feat[:, -1, :]
        pred = self.pred_head(last_step)
        return pred
