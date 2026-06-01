"""
评估与解释模块
"""
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap

def evaluate_model(model, test_loader, scaler_info, target_stock_id, device):
    """在测试集上评估模型，返回真实价格、预测价格和评估指标。"""
    model.eval()
    all_preds, all_targets, all_ids = [], [], []

    with torch.no_grad():
        for X_num, X_text, y, ids in test_loader:
            X_num, X_text = X_num.to(device), X_text.to(device)
            pred, _ = model(X_num, X_text)
            pred = pred.squeeze().cpu().numpy()
            all_preds.append(pred)
            all_targets.append(y.numpy())
            all_ids.extend(ids)

    preds_flat = np.concatenate(all_preds)
    targets_flat = np.concatenate(all_targets)
    ids_flat = np.array(all_ids)

    mask = ids_flat == target_stock_id
    if np.sum(mask) == 0:
        raise ValueError(f"测试集中无目标股票 {target_stock_id} 的数据。")

    p_norm = preds_flat[mask]
    t_norm = targets_flat[mask]

    close_min = scaler_info['close_min']
    close_range = scaler_info['close_range']
    real_price = t_norm * close_range + close_min
    pred_price = p_norm * close_range + close_min

    rmse = np.sqrt(mean_squared_error(real_price, pred_price))
    mae = mean_absolute_error(real_price, pred_price)
    mape = np.mean(np.abs((real_price - pred_price) / real_price)) * 100

    n_days = len(real_price)
    if n_days > 1 and real_price[0] != 0:
        total_return = real_price[-1] / real_price[0]
        annualized_return = total_return ** (252 / n_days) - 1
    else:
        annualized_return = 0.0

    # 夏普比率
    if n_days > 1:
        daily_returns = np.diff(pred_price) / pred_price[:-1]
        std_returns = np.std(daily_returns)
        if std_returns > 0:
            sharpe_ratio = np.mean(daily_returns) / std_returns * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    metrics = {
        'RMSE': rmse, 'MAE': mae, 'MAPE': mape,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio
    }
    return real_price, pred_price, metrics


def explain_with_shap(model, X_num_sample, X_text_sample, device,
                      feature_names_num=None, background_size=50):
    model.eval()

    class SHAPModelWrapper(torch.nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.model = original_model

        def forward(self, x_num, x_text):
            pred, _ = self.model(x_num, x_text)
            return pred

    wrapper = SHAPModelWrapper(model).to(device)
    wrapper.eval()

    n_samples = X_num_sample.shape[0]
    bg_size = min(background_size, n_samples)
    bg_num = torch.tensor(X_num_sample[:bg_size], dtype=torch.float32, requires_grad=True).to(device)
    bg_text = torch.tensor(X_text_sample[:bg_size], dtype=torch.float32, requires_grad=True).to(device)

    explainer = shap.GradientExplainer(wrapper, [bg_num, bg_text])

    test_num = torch.tensor(X_num_sample, dtype=torch.float32, requires_grad=True).to(device)
    test_text = torch.tensor(X_text_sample, dtype=torch.float32, requires_grad=True).to(device)

    shap_values = explainer.shap_values([test_num, test_text])

    # 兼容新旧版本 SHAP 返回值格式
    if isinstance(shap_values, list) and len(shap_values) > 0:
        if isinstance(shap_values[0], list):
            shap_num = np.array(shap_values[0][0])
            shap_text = np.array(shap_values[0][1])
        else:
            shap_num = np.array(shap_values[0])
            shap_text = np.array(shap_values[1])
    else:
        raise ValueError(f"Unexpected SHAP values format: {type(shap_values)}")

    # 若因输出维度导致多出轴（如 (n, 1, seq, feat) 或 (n, seq, feat, 1)），则 squeeze
    # 注意：只 squeeze 一个 size==1 的轴，避免 n_samples=1 时过度降维
    def _squeeze_singleton(arr, name):
        if arr.ndim == 4:
            if arr.shape[1] == 1:
                return arr.squeeze(1)
            elif arr.shape[-1] == 1:
                return arr.squeeze(-1)
            else:
                # 查找任意 size==1 的轴，只移除一个以降到 3 维
                singleton_axes = [i for i, s in enumerate(arr.shape) if s == 1]
                if singleton_axes:
                    return np.squeeze(arr, axis=singleton_axes[0])
                else:
                    raise ValueError(
                        f"{name} 为4维但没有 size==1 的轴，无法 squeeze: shape={arr.shape}"
                    )
        return arr

    shap_num = _squeeze_singleton(shap_num, 'shap_num')
    shap_text = _squeeze_singleton(shap_text, 'shap_text')

    # 导出重要性指标
    num_importance = np.abs(shap_num).mean(axis=(0, 1))      # (num_feat,)
    text_temporal_per_sample = np.abs(shap_text).sum(axis=-1)  # (n_samples, seq_len)
    text_temporal_mean = text_temporal_per_sample.mean(axis=0)

    if feature_names_num is None:
        feature_names_num = [f"feat_{i}" for i in range(shap_num.shape[-1])]

    return {
        'shap_values_num': shap_num,
        'shap_values_text': shap_text,
        'num_feature_importance': num_importance,
        'text_temporal_importance_per_sample': text_temporal_per_sample,
        'text_temporal_importance_mean': text_temporal_mean,
        'feature_names_num': feature_names_num
    }