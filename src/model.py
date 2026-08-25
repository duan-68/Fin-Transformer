"""
模型定义模块
"""

import math
import torch
import torch.nn as nn
import config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class FinancialPriorBias(nn.Module):
    """从数值特征中提取波动率和成交量信号，生成注意力偏置"""
    def __init__(self, num_input_dim, nhead):
        super().__init__()
        self.nhead = nhead
        self.bias_net = nn.Sequential(
            nn.Linear(num_input_dim, nhead),
            nn.Tanh()
        )
        self.scale = nn.Parameter(torch.ones(1) * config.PRIOR_BIAS_SCALE)

    def forward(self, x_num_raw):
        # x_num_raw: [B, L, num_input_dim]
        B, L, _ = x_num_raw.shape
        bias = self.bias_net(x_num_raw)       # [B, L, nhead]
        bias = bias * self.scale              # 缩放控制先验强度
        bias = bias.permute(0, 2, 1)          # [B, nhead, L]
        bias = bias.unsqueeze(2).expand(-1, -1, L, -1)  # [B, nhead, L, L]
        return bias.reshape(B * self.nhead, L, L)


class FinTransformerModel(nn.Module):
    """
    双流融合模型：
    - 数值流：Transformer 编码器
    - 文本流：线性投影 + Transformer 编码器
    - 融合：交叉注意力 (Q=num, K/V=text) + 残差
    - 预测头：全连接层
    """
    def __init__(self, num_input_dim, text_input_dim=768, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1, pred_length=1):
        super().__init__()

        # 数值流
        self.num_proj = nn.Linear(num_input_dim, d_model)
        self.num_pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.num_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # 文本流
        self.text_proj = nn.Linear(text_input_dim, d_model)
        self.text_pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers
        )

        # 交叉注意力融合
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.prior_bias = FinancialPriorBias(num_input_dim, nhead)
        self.fusion_norm = nn.LayerNorm(d_model)

        # 预测头
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, pred_length)
        )

    def forward(self, x_num, x_text):
        num_feat = self.num_proj(x_num) * math.sqrt(x_num.size(-1))
        num_feat = self.num_pos_encoder(num_feat)
        num_feat = self.num_encoder(num_feat)

        text_feat = self.text_proj(x_text)
        text_feat = self.text_pos_encoder(text_feat)
        text_feat = self.text_encoder(text_feat)

        attn_bias = self.prior_bias(x_num)
        cross_out, attn_weights = self.cross_attn(num_feat, text_feat, text_feat, attn_mask=attn_bias)

        num_encoded_last = num_feat[:, -1, :]
        cross_attn_output = cross_out[:, -1, :]
        fused = self.fusion_norm(num_encoded_last + cross_attn_output)

        out = self.pred_head(fused)
        return out, attn_weights


class SimpleTransformerModel(nn.Module):
    """纯数值单流 Transformer 基线模型

    仅使用数值特征进行时序预测，不含文本流和交叉注意力机制。
    """

    def __init__(self, num_input_dim=7, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1, pred_length=1):
        super().__init__()
        self.num_proj = nn.Linear(num_input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pred_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, pred_length),
        )

    def forward(self, x_num):
        feat = self.num_proj(x_num) * math.sqrt(self.num_proj.out_features)
        feat = self.pos_encoder(feat)
        feat = self.encoder(feat)
        last_step = feat[:, -1, :]
        out = self.pred_head(last_step)
        return out


class ConcatFusionModel(nn.Module):
    """双流编码 + 拼接融合模型（消融实验用）

    数值流和文本流分别用 Transformer 编码后，拼接最后时间步输出通过线性层融合。
    不含交叉注意力和金融先验偏置。
    """

    def __init__(self, num_input_dim=7, text_input_dim=768, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1, pred_length=1):
        super().__init__()

        self.num_proj = nn.Linear(num_input_dim, d_model)
        self.num_pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.num_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.text_proj = nn.Linear(text_input_dim, d_model)
        self.text_pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers
        )

        self.fusion_fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU()
        )

        self.pred_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, pred_length)
        )

    def forward(self, x_num, x_text):
        num_feat = self.num_proj(x_num) * math.sqrt(self.num_proj.out_features)
        num_feat = self.num_pos_encoder(num_feat)
        num_feat = self.num_encoder(num_feat)

        text_feat = self.text_proj(x_text)
        text_feat = self.text_pos_encoder(text_feat)
        text_feat = self.text_encoder(text_feat)

        num_last = num_feat[:, -1, :]
        text_last = text_feat[:, -1, :]
        concat_feat = torch.cat([num_last, text_last], dim=-1)

        fused = self.fusion_fc(concat_feat)
        out = self.pred_head(fused)
        return out, None


class CrossAttnOnlyModel(nn.Module):
    """双流编码 + 交叉注意力模型（消融实验用）

    使用交叉注意力融合数值流和文本流，不含金融先验偏置。
    """

    def __init__(self, num_input_dim=7, text_input_dim=768, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1, pred_length=1):
        super().__init__()

        self.num_proj = nn.Linear(num_input_dim, d_model)
        self.num_pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.num_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.text_proj = nn.Linear(text_input_dim, d_model)
        self.text_pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers
        )

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.fusion_norm = nn.LayerNorm(d_model)

        self.pred_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, pred_length)
        )

    def forward(self, x_num, x_text):
        num_feat = self.num_proj(x_num) * math.sqrt(self.num_proj.out_features)
        num_feat = self.num_pos_encoder(num_feat)
        num_feat = self.num_encoder(num_feat)

        text_feat = self.text_proj(x_text)
        text_feat = self.text_pos_encoder(text_feat)
        text_feat = self.text_encoder(text_feat)

        fused, attn_weights = self.cross_attn(num_feat, text_feat, text_feat, attn_mask=None)
        fused = self.fusion_norm(num_feat + fused)

        last_step = fused[:, -1, :]
        out = self.pred_head(last_step)
        return out, attn_weights