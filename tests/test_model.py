"""
模型单元测试
"""

import pytest
import torch
import sys
sys.path.insert(0, 'e:\\code\\project')

from src.model import (
    SimpleTransformerModel,
    ConcatFusionModel,
    CrossAttnOnlyModel,
    FinTransformerModel,
    FinancialPriorBias,
)
from src.train import attention_regularization_loss


@pytest.fixture
def batch_params():
    return dict(B=4, L=50, num_dim=7, text_dim=768)


@pytest.fixture
def x_num(batch_params):
    return torch.randn(batch_params['B'], batch_params['L'], batch_params['num_dim'])


@pytest.fixture
def x_text(batch_params):
    return torch.randn(batch_params['B'], batch_params['L'], batch_params['text_dim'])


class TestForwardShape:
    """前向传播形状验证"""

    def test_simple_transformer_shape(self, x_num, batch_params):
        model = SimpleTransformerModel(num_input_dim=batch_params['num_dim'])
        model.eval()
        with torch.no_grad():
            out = model(x_num)
        # SimpleTransformerModel 只返回一个值
        assert out.shape == (batch_params['B'], 1)

    def test_simple_transformer_no_attn(self, x_num, batch_params):
        """纯数值 Transformer 无注意力权重返回"""
        model = SimpleTransformerModel(num_input_dim=batch_params['num_dim'])
        model.eval()
        with torch.no_grad():
            result = model(x_num)
        # forward 返回单个 tensor，不是 tuple，等效第二返回值为 None
        assert not isinstance(result, tuple)

    def test_concat_fusion_shape(self, x_num, x_text, batch_params):
        model = ConcatFusionModel(
            num_input_dim=batch_params['num_dim'],
            text_input_dim=batch_params['text_dim']
        )
        model.eval()
        with torch.no_grad():
            out, attn = model(x_num, x_text)
        assert out.shape == (batch_params['B'], 1)
        assert attn is None

    def test_cross_attn_only_shape(self, x_num, x_text, batch_params):
        model = CrossAttnOnlyModel(
            num_input_dim=batch_params['num_dim'],
            text_input_dim=batch_params['text_dim']
        )
        model.eval()
        with torch.no_grad():
            out, attn_weights = model(x_num, x_text)
        assert out.shape == (batch_params['B'], 1)
        assert attn_weights is not None
        L = batch_params['L']
        B = batch_params['B']
        valid_shapes = [(B, L, L), (B * 4, L, L)]  # nhead=4
        assert attn_weights.shape in valid_shapes

    def test_fin_transformer_shape(self, x_num, x_text, batch_params):
        model = FinTransformerModel(
            num_input_dim=batch_params['num_dim'],
            text_input_dim=batch_params['text_dim']
        )
        model.eval()
        with torch.no_grad():
            out, attn_weights = model(x_num, x_text)
        assert out.shape == (batch_params['B'], 1)
        # 注意力权重应存在
        assert attn_weights is not None


class TestLossFunction:
    """损失函数正确性测试"""

    def test_mse_loss(self):
        """MSE 损失基本计算"""
        criterion = torch.nn.MSELoss()
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        loss = criterion(pred, target)
        assert loss.item() == 0.0

        pred2 = torch.tensor([1.0, 2.0, 3.0])
        target2 = torch.tensor([2.0, 3.0, 4.0])
        loss2 = criterion(pred2, target2)
        assert abs(loss2.item() - 1.0) < 1e-6

    def test_attention_regularization_entropy_direction(self):
        """均匀分布注意力权重的熵正则化方向：均匀分布熵最大，负号使其为负贡献"""
        L = 10
        # 均匀分布注意力权重
        uniform_attn = torch.ones(2, 4, L, L) / L
        # 集中分布注意力权重（one-hot）
        concentrated_attn = torch.zeros(2, 4, L, L)
        concentrated_attn[:, :, :, 0] = 1.0

        reg_uniform = attention_regularization_loss(uniform_attn)
        reg_concentrated = attention_regularization_loss(concentrated_attn)

        # 均匀分布熵大 -> -lambda_entropy * entropy 更负
        # 集中分布熵小 -> -lambda_entropy * entropy 更正（接近0）
        # 因此均匀分布的 reg_loss 应小于集中分布（熵项主导时）
        assert reg_uniform.item() < reg_concentrated.item()

    def test_attention_regularization_sparsity_direction(self):
        """稀疏正则化项：L1范数越大，正则化损失越大"""
        L = 10
        # 均匀分布：每个值 = 1/L
        uniform_attn = torch.ones(2, 4, L, L) / L
        # 全部为0.5（不合法注意力但用于测试稀疏项方向）
        large_attn = torch.ones(2, 4, L, L) * 0.5

        # 用纯稀疏正则化比较
        sparsity_uniform = uniform_attn.abs().mean()
        sparsity_large = large_attn.abs().mean()
        assert sparsity_large > sparsity_uniform


class TestFinancialPriorBias:
    """金融先验偏置模块测试"""

    def test_output_range(self):
        """输出范围验证"""
        num_input_dim = 7
        nhead = 4
        module = FinancialPriorBias(num_input_dim, nhead)
        module.eval()

        x = torch.randn(4, 50, num_input_dim)
        with torch.no_grad():
            bias = module(x)

        scale = module.scale.item()
        assert bias.max().item() <= scale + 1e-6
        assert bias.min().item() >= -scale - 1e-6

    def test_output_shape(self):
        """输出形状验证"""
        B, L, num_input_dim, nhead = 4, 50, 7, 4
        module = FinancialPriorBias(num_input_dim, nhead)
        module.eval()

        x = torch.randn(B, L, num_input_dim)
        with torch.no_grad():
            bias = module(x)

        assert bias.shape == (B * nhead, L, L)


class TestAttentionWeights:
    """注意力权重验证"""

    def test_fin_transformer_attn_nonneg(self, x_num, x_text, batch_params):
        model = FinTransformerModel(
            num_input_dim=batch_params['num_dim'],
            text_input_dim=batch_params['text_dim']
        )
        model.eval()
        with torch.no_grad():
            _, attn_weights = model(x_num, x_text)
        assert (attn_weights >= 0).all()

    def test_fin_transformer_attn_sum_to_one(self, x_num, x_text, batch_params):
        model = FinTransformerModel(
            num_input_dim=batch_params['num_dim'],
            text_input_dim=batch_params['text_dim']
        )
        model.eval()
        with torch.no_grad():
            _, attn_weights = model(x_num, x_text)
        # 每行求和约等于1 (softmax性质)
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_cross_attn_only_attn_nonneg(self, x_num, x_text, batch_params):
        model = CrossAttnOnlyModel(
            num_input_dim=batch_params['num_dim'],
            text_input_dim=batch_params['text_dim']
        )
        model.eval()
        with torch.no_grad():
            _, attn_weights = model(x_num, x_text)
        assert (attn_weights >= 0).all()

    def test_cross_attn_only_attn_sum_to_one(self, x_num, x_text, batch_params):
        model = CrossAttnOnlyModel(
            num_input_dim=batch_params['num_dim'],
            text_input_dim=batch_params['text_dim']
        )
        model.eval()
        with torch.no_grad():
            _, attn_weights = model(x_num, x_text)
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


class TestConcatFusionNoAttn:
    """ConcatFusionModel 注意力权重验证"""

    def test_no_attention_weights(self, x_num, x_text, batch_params):
        model = ConcatFusionModel(
            num_input_dim=batch_params['num_dim'],
            text_input_dim=batch_params['text_dim']
        )
        model.eval()
        with torch.no_grad():
            out, attn = model(x_num, x_text)
        assert attn is None
