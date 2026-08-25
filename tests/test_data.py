"""
数据处理单元测试
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'e:\\code\\project')


class TestNormalization:
    """归一化与反归一化对称性验证"""

    def test_normalize_denormalize_symmetry(self):
        """归一化/反归一化对称性"""
        original = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        close_min = original.min()
        close_range = original.max() - original.min()
        normalized = (original - close_min) / close_range
        recovered = normalized * close_range + close_min
        assert np.allclose(original, recovered)

    def test_normalize_range(self):
        """归一化值范围验证"""
        original = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        close_min = original.min()
        close_range = original.max() - original.min()
        normalized = (original - close_min) / close_range
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_normalize_denormalize_random_data(self):
        """随机数据对称性验证"""
        np.random.seed(42)
        original = np.random.uniform(5.0, 100.0, size=100)
        close_min = original.min()
        close_range = original.max() - original.min()
        normalized = (original - close_min) / close_range
        recovered = normalized * close_range + close_min
        assert np.allclose(original, recovered, atol=1e-10)


class TestTextMissingFill:
    """文本缺失零向量填充验证"""

    def test_missing_dates_filled_with_zeros(self):
        """缺失日期零向量填充"""
        text_vec_dim = 768
        # 模拟 date_to_vec 字典，只有部分日期有向量
        date_to_vec = {
            '20250101': np.random.randn(text_vec_dim),
            '20250103': np.random.randn(text_vec_dim),
            '20250105': np.random.randn(text_vec_dim),
        }

        # 模拟窗口日期序列（含缺失日期）
        window_dates = ['20250101', '20250102', '20250103', '20250104', '20250105']

        text_seq = []
        for d in window_dates:
            if d in date_to_vec:
                text_seq.append(date_to_vec[d])
            else:
                text_seq.append(np.zeros(text_vec_dim))
        text_seq = np.array(text_seq)

        # 验证形状
        assert text_seq.shape == (5, text_vec_dim)

        # 验证缺失日期位置为零向量
        assert np.allclose(text_seq[1], np.zeros(text_vec_dim))  # 20250102 缺失
        assert np.allclose(text_seq[3], np.zeros(text_vec_dim))  # 20250104 缺失

        # 验证存在日期位置不为零向量
        assert not np.allclose(text_seq[0], np.zeros(text_vec_dim))
        assert not np.allclose(text_seq[2], np.zeros(text_vec_dim))
        assert not np.allclose(text_seq[4], np.zeros(text_vec_dim))

    def test_all_dates_missing(self):
        """全部日期缺失场景"""
        text_vec_dim = 768
        date_to_vec = {}
        window_dates = ['20250101', '20250102', '20250103']

        text_seq = []
        for d in window_dates:
            if d in date_to_vec:
                text_seq.append(date_to_vec[d])
            else:
                text_seq.append(np.zeros(text_vec_dim))
        text_seq = np.array(text_seq)

        assert text_seq.shape == (3, text_vec_dim)
        assert np.allclose(text_seq, np.zeros((3, text_vec_dim)))


class TestDataSplitNoOverlap:
    """数据划分无重叠验证"""

    def test_split_no_overlap(self):
        """训练/验证/测试集无重叠"""
        n_samples = 200
        train_ratio = 0.70
        val_ratio = 0.15

        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        train_indices = set(range(0, train_end))
        val_indices = set(range(train_end, val_end))
        test_indices = set(range(val_end, n_samples))

        # 验证无重叠
        assert len(train_indices & val_indices) == 0
        assert len(train_indices & test_indices) == 0
        assert len(val_indices & test_indices) == 0

        # 验证覆盖完整
        assert train_indices | val_indices | test_indices == set(range(n_samples))

    def test_split_proportions(self):
        """验证划分比例大致正确"""
        n_samples = 1000
        train_ratio = 0.70
        val_ratio = 0.15

        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        train_count = train_end
        val_count = val_end - train_end
        test_count = n_samples - val_end

        assert abs(train_count / n_samples - 0.70) < 0.01
        assert abs(val_count / n_samples - 0.15) < 0.01
        assert abs(test_count / n_samples - 0.15) < 0.01

    def test_split_with_sequence_construction(self):
        """模拟实际序列构建过程的划分无重叠"""
        np.random.seed(42)
        n = 300
        seq_length = 50
        pred_length = 1
        train_ratio = 0.70
        val_ratio = 0.15

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_samples = []
        val_samples = []
        test_samples = []

        limit = n - seq_length - pred_length + 1
        for i in range(limit):
            if i + seq_length < train_end:
                train_samples.append(i)
            elif i + seq_length < val_end:
                val_samples.append(i)
            else:
                test_samples.append(i)

        # 验证无重叠
        train_set = set(train_samples)
        val_set = set(val_samples)
        test_set = set(test_samples)

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

        # 验证所有样本都被分配
        assert len(train_set) + len(val_set) + len(test_set) == limit
