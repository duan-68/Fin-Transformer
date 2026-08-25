#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股价预测脚本：使用已训练的 Fin-Transformer 模型预测下一交易日收盘价和涨跌幅。
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from pymongo import MongoClient

# 确保项目根目录在 sys.path 中
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import config
from src.model import FinTransformerModel
from src.data import run_finbert_inference, aggregate_daily_text_vectors
from src.visualize import plot_prediction

# ======================== 配置区域 ========================
DB_NAME_TEST = "posts_test"  # 使用2026年新文本数据库
PRICE_DIR = os.path.join(ROOT_DIR, "data", "data", "prices")          # 2025年原始价格（仅用于scaler重建）
PRICE_TEST_DIR = os.path.join(ROOT_DIR, "data", "data", "prices_test") # 2026年新价格
PREDICT_OUTPUT_DIR = os.path.join(ROOT_DIR, "reports", "predict")
PREDICT_SENTIMENT_DIR = os.path.join(PREDICT_OUTPUT_DIR, "sentiment")
PREDICT_EMBEDDINGS_DIR = os.path.join(PREDICT_OUTPUT_DIR, "sentiment", "embeddings")
PREDICT_FIGURES_DIR = os.path.join(PREDICT_OUTPUT_DIR, "figures")
# ==========================================================


def get_next_trade_date_label(last_date):
    """根据最后一个交易日获取下一个交易日的日期标签。"""
    try:
        from data.get_data.get_price import _get_pro
        pro = _get_pro()
        last_date_str_cal = last_date.strftime("%Y%m%d")
        future_end = (last_date + timedelta(days=30)).strftime("%Y%m%d")
        df_cal = pro.trade_cal(exchange='SSE', start_date=last_date_str_cal, end_date=future_end)
        future_trade_days = df_cal[(df_cal['is_open'] == 1) & (df_cal['cal_date'] > last_date_str_cal)]['cal_date'].sort_values().tolist()
        if future_trade_days:
            next_trade_date = datetime.strptime(future_trade_days[0], "%Y%m%d")
            return next_trade_date.strftime('%Y-%m-%d') + '(下一交易日)'
        else:
            return '下一交易日'
    except Exception:
        return '下一交易日'


def rebuild_scaler(stock_code):
    """从训练数据重建 MinMaxScaler，确保与训练时一致。

    返回: (scaler, scaler_info)
    """
    price_path = os.path.join(PRICE_DIR, f"{stock_code}.csv")
    df = pd.read_csv(price_path)
    feature_cols = config.FEATURE_COLS

    scaler = MinMaxScaler()
    scaler.fit(df[feature_cols].values)

    close_idx = feature_cols.index(config.TARGET_COL)
    scaler_info = {
        'close_min': scaler.data_min_[close_idx],
        'close_range': scaler.data_range_[close_idx]
    }

    return scaler, scaler_info


def prepare_price_sequence(stock_code, scaler):
    """准备价格数值序列，取最后90个日历天的交易数据。

    返回: (all_norm, all_dates, all_close)
    """
    feature_cols = config.FEATURE_COLS

    price_path = os.path.join(PRICE_TEST_DIR, f"{stock_code}.csv")
    df_all = pd.read_csv(price_path)
    df_all = df_all.sort_values('trade_date').reset_index(drop=True)

    last_date = str(df_all['trade_date'].iloc[-1])
    last_dt = datetime.strptime(last_date, "%Y%m%d")
    cutoff_dt = last_dt - timedelta(days=90)
    cutoff_date = int(cutoff_dt.strftime("%Y%m%d"))

    df_new = df_all[df_all['trade_date'] >= cutoff_date].reset_index(drop=True)

    n_new = len(df_new)
    if n_new < config.SEQ_LENGTH:
        print(f"    警告: 最后90天数据仅有 {n_new} 个交易日，将使用全部可用数据")
    else:
        print(f"    最后90日历天共有 {n_new} 个交易日")

    df_combined = df_new.reset_index(drop=True)

    all_features = df_combined[feature_cols].values
    all_norm = scaler.transform(all_features)

    all_dates = df_combined['trade_date'].astype(str).tolist()
    all_close = df_combined['close'].values

    return all_norm, all_dates, all_close


def prepare_text_sequence(stock_code, trading_dates, price_df_for_agg, progress_callback=None):
    """准备文本向量序列，若无文本数据则返回零向量。"""
    n_days = len(trading_dates)

    # 检查MongoDB中是否有该股票的文本数据
    client = MongoClient(config.MONGO_HOST, config.MONGO_PORT)
    db = client[DB_NAME_TEST]
    collection = db[f"post_{stock_code}"]
    doc_count = collection.count_documents({})

    if doc_count == 0:
        client.close()
        print(f"    股票 {stock_code}: {DB_NAME_TEST} 中无文本数据，使用零向量")
        return np.zeros((n_days, 768), dtype=np.float32), False

    # 计算日期范围：第一个交易日前一天 ~ 最后一个交易日（或当前时间，取较大值）
    first_dt = datetime.strptime(str(trading_dates[0]), "%Y%m%d") - timedelta(days=1)
    date_start = first_dt.strftime("%Y%m%d") + "0000"
    # 若当前时间晚于最后一个交易日，扩展查询范围以包含周末帖子
    now = datetime.now()
    last_trade_dt = datetime.strptime(str(trading_dates[-1]), "%Y%m%d")
    if now.date() > last_trade_dt.date():
        date_end = now.strftime("%Y%m%d") + now.strftime("%H%M")
    else:
        date_end = str(trading_dates[-1]) + "2359"

    # 带日期过滤查询实际符合条件的帖子数
    query_filter = {'post_date': {'$gte': date_start, '$lte': date_end}}
    doc_count_filtered = collection.count_documents(query_filter)
    client.close()

    print(f"    股票 {stock_code}: {DB_NAME_TEST} 中共 {doc_count} 条文本数据，日期范围 {date_start}~{date_end} 内 {doc_count_filtered} 条")

    # --- 运行 FinBERT 推理 ---
    # 检查是否已有结果（避免重复运行）
    sentiment_csv = os.path.join(PREDICT_SENTIMENT_DIR, f"{stock_code}_sentiment_results.csv")
    emb_dir = os.path.join(PREDICT_EMBEDDINGS_DIR, stock_code)

    if os.path.exists(sentiment_csv):
        # 检查数据充足性，防止使用残留的不完整数据
        try:
            row_count = sum(1 for _ in open(sentiment_csv, encoding='utf-8')) - 1  # 减去表头
        except Exception:
            row_count = 0
        if row_count <= 10:
            print(f"    已有情感分析结果但数据不足（仅{row_count}行），删除后重新运行FinBERT")
            os.remove(sentiment_csv)
        else:
            print(f"    已有情感分析结果（{row_count}行），跳过FinBERT推理")

    if not os.path.exists(sentiment_csv):
        print(f"    运行FinBERT推理...")
        # 需要临时修改输出路径以避免覆盖训练结果
        # 保存原始路径
        orig_sentiment_dir = config.SENTIMENT_DIR
        orig_embeddings_dir = config.EMBEDDINGS_DIR

        # 临时切换到预测输出目录
        config.SENTIMENT_DIR = PREDICT_SENTIMENT_DIR
        config.EMBEDDINGS_DIR = PREDICT_EMBEDDINGS_DIR

        try:
            run_finbert_inference(stock_code, db_name=DB_NAME_TEST,
                                  date_start=date_start, date_end=date_end,
                                  progress_callback=progress_callback)
        finally:
            # 恢复原始路径
            config.SENTIMENT_DIR = orig_sentiment_dir
            config.EMBEDDINGS_DIR = orig_embeddings_dir

    # --- 聚合每日文本向量 ---
    # 读取情感分析结果CSV
    df_sent = pd.read_csv(sentiment_csv)

    # 构建datetime列用于时间窗口
    df_sent['datetime'] = pd.to_datetime(
        df_sent['post_date'].astype(str) + ' ' + df_sent['post_time'],
        format='%Y%m%d %H:%M',
        errors='coerce'
    )
    df_sent = df_sent.dropna(subset=['datetime'])

    # 按交易日聚合（时间窗口：前一交易日15:00 → 当日09:30，最后一个交易日可扩展至当前时间）
    daily_vectors = []
    for i in range(n_days):
        date_str = trading_dates[i]
        day_dt = datetime.strptime(str(date_str), "%Y%m%d")

        if i == n_days - 1 and now.date() > day_dt.date():
            # 最后一个交易日且当前时间在该交易日之后（如周末）：扩展窗口到当前时间
            if i > 0:
                prev_date_str = trading_dates[i-1]
                prev_dt = datetime.strptime(str(prev_date_str), "%Y%m%d")
                window_start = datetime(prev_dt.year, prev_dt.month, prev_dt.day, 15, 0)
            else:
                window_start = datetime(day_dt.year, day_dt.month, day_dt.day, 9, 30) - timedelta(hours=18)
            window_end = now  # 扩展到当前时间，纳入周末帖子
            print(f"  最后交易日 {date_str}: 文本窗口扩展至 {now.strftime('%Y-%m-%d %H:%M')}（含周末帖子）")
        else:
            # 标准窗口：前一交易日15:00 ~ 当日09:30
            window_end = datetime(day_dt.year, day_dt.month, day_dt.day, 9, 30)

            # 找到前一个交易日
            if i > 0:
                prev_date_str = trading_dates[i-1]
                prev_dt = datetime.strptime(str(prev_date_str), "%Y%m%d")
                window_start = datetime(prev_dt.year, prev_dt.month, prev_dt.day, 15, 0)
            else:
                window_start = window_end - timedelta(hours=18)

        df_window = df_sent[
            (df_sent['datetime'] >= window_start) &
            (df_sent['datetime'] < window_end)
        ]

        if len(df_window) == 0:
            daily_vectors.append(np.zeros(768, dtype=np.float32))
        else:
            # 置信度加权平均
            embeddings = []
            confidences = []
            for _, row in df_window.iterrows():
                emb_path = os.path.join(PREDICT_EMBEDDINGS_DIR, stock_code, f"{row['_id']}.npy")
                if os.path.exists(emb_path):
                    emb = np.load(emb_path)
                    embeddings.append(emb)
                    confidences.append(row['confidence'])

            if embeddings:
                embeddings = np.array(embeddings)
                confidences = np.array(confidences)
                weighted_sum = np.sum(embeddings * confidences[:, np.newaxis], axis=0)
                daily_vec = weighted_sum / np.sum(confidences)
                daily_vectors.append(daily_vec.astype(np.float32))
            else:
                daily_vectors.append(np.zeros(768, dtype=np.float32))

    text_vecs = np.array(daily_vectors)
    return text_vecs, True


def predict_stock(stock_info):
    """对单只股票执行完整预测流程，返回预测结果字典，失败返回 None。"""
    stock_code = stock_info['code']
    stock_name = stock_info['name']
    stock_suffix = stock_info['suffix']

    print(f"\n{'='*60}")
    print(f"  预测股票: {stock_code} {stock_name}")
    print(f"{'='*60}")

    try:
        # 检查模型文件是否存在
        model_path = config.get_model_save_path(stock_code)
        if not os.path.exists(model_path):
            print(f"  模型文件不存在: {model_path}，跳过")
            return None

        # 检查价格数据是否存在
        price_path = os.path.join(PRICE_TEST_DIR, f"{stock_code}.csv")
        if not os.path.exists(price_path):
            print(f"  价格数据不存在: {price_path}，跳过")
            return None

        # 步骤1：重建Scaler
        print(f"  [1/6] 重建训练数据Scaler...")
        scaler, scaler_info = rebuild_scaler(stock_code)
        print(f"    close_min={scaler_info['close_min']:.4f}, close_range={scaler_info['close_range']:.4f}")

        # 步骤2：准备价格序列
        print(f"  [2/6] 准备价格序列...")
        all_norm, all_dates, all_close = prepare_price_sequence(stock_code, scaler)
        print(f"    共 {len(all_dates)} 个交易日，日期范围: {all_dates[0]} ~ {all_dates[-1]}")

        print(f"  [3/6] 准备文本序列...")
        text_vecs, has_text = prepare_text_sequence(stock_code, all_dates, None)

        # 步骤4：模型加载与回测推理
        print(f"  [4/6] 加载模型并执行回测推理...")
        device = torch.device(config.DEVICE)

        model = FinTransformerModel(
            num_input_dim=len(config.FEATURE_COLS),
            text_input_dim=768,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_layers=config.NUM_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            pred_length=config.PRED_LENGTH
        ).to(device)

        state_dict = torch.load(model_path, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"    警告: 模型权重缺失以下key（将使用默认值）: {missing}")
        model.eval()

        seq_len = config.SEQ_LENGTH  # 50
        n_total = len(all_norm)

        # 回测：滑动窗口生成预测
        backtest_real = []
        backtest_pred = []
        backtest_dates = []

        # 从序列中创建所有可能的窗口
        # 窗口 i: all_norm[i:i+50] → 预测 all_norm[i+50] 的close
        for i in range(n_total - seq_len):
            x_num_np = all_norm[i : i + seq_len]  # (50, 7)
            x_text_np = text_vecs[i : i + seq_len]  # (50, 768)

            x_num_t = torch.tensor(x_num_np, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 50, 7)
            x_text_t = torch.tensor(x_text_np, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 50, 768)

            with torch.no_grad():
                pred_norm, _ = model(x_num_t, x_text_t)

            pred_price = pred_norm.item() * scaler_info['close_range'] + scaler_info['close_min']
            real_price = all_close[i + seq_len]

            backtest_pred.append(pred_price)
            backtest_real.append(real_price)
            backtest_dates.append(all_dates[i + seq_len])

        # 下一交易日预测：使用最后50天
        x_num_last = all_norm[-seq_len:]  # (50, 7)
        x_text_last = text_vecs[-seq_len:]  # (50, 768)

        x_num_t = torch.tensor(x_num_last, dtype=torch.float32).unsqueeze(0).to(device)
        x_text_t = torch.tensor(x_text_last, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_norm_next, _ = model(x_num_t, x_text_t)

        pred_next = pred_norm_next.item() * scaler_info['close_range'] + scaler_info['close_min']
        last_close = all_close[-1]
        change = pred_next - last_close
        pct_change = change / last_close * 100

        # 步骤5：计算回测指标
        print(f"  [5/6] 计算回测指标...")
        backtest_real = np.array(backtest_real)
        backtest_pred = np.array(backtest_pred)

        if len(backtest_real) > 0:
            rmse = np.sqrt(np.mean((backtest_real - backtest_pred) ** 2))
            mae = np.mean(np.abs(backtest_real - backtest_pred))
            mape = np.mean(np.abs((backtest_real - backtest_pred) / backtest_real)) * 100

            # 年化收益率（基于真实价格序列）
            n_days = len(backtest_real)
            if n_days > 1 and backtest_real[0] != 0:
                total_return = backtest_real[-1] / backtest_real[0]
                annualized_return = total_return ** (252 / n_days) - 1
            else:
                annualized_return = 0.0

            # 夏普比率（基于预测价格的日收益率，假设无风险利率为0）
            if len(backtest_pred) > 1:
                daily_returns = np.diff(backtest_pred) / backtest_pred[:-1]
                std_returns = np.std(daily_returns)
                if std_returns > 0:
                    sharpe_ratio = np.mean(daily_returns) / std_returns * np.sqrt(252)
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0

            metrics = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape,
                       'annualized_return': annualized_return, 'sharpe_ratio': sharpe_ratio}
        else:
            metrics = {'RMSE': 0, 'MAE': 0, 'MAPE': 0,
                       'annualized_return': 0.0, 'sharpe_ratio': 0.0}

        print(f"    回测样本数: {len(backtest_real)}")
        if len(backtest_real) > 0:
            print(f"    RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%")
            print(f"    年化收益率={metrics['annualized_return']*100:.2f}%, 夏普比率={metrics['sharpe_ratio']:.4f}")
        else:
            print(f"    警告: 交易日不足{config.SEQ_LENGTH+1}个，无法产生回测点，跳过可视化")

        # 步骤6：可视化
        print(f"  [6/6] 生成预测可视化...")

        fig_path = os.path.join(PREDICT_FIGURES_DIR, f"predict_{stock_code}.png")

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        if len(backtest_real) > 0:
            fig, ax = plt.subplots(figsize=(14, 6))

            n_show = min(7, len(backtest_real))
            plot_real = backtest_real[-n_show:]
            plot_pred = backtest_pred[-n_show:]
            plot_dates = backtest_dates[-n_show:]

            x_idx = range(n_show)

            ax.plot(x_idx, plot_real, 'b-', linewidth=1.5, label='实际收盘价', alpha=0.9)
            ax.plot(x_idx, plot_pred, 'r--', linewidth=1.5, label='模型预测收盘价', alpha=0.9)

            next_x = n_show
            ax.plot([next_x - 1, next_x], [plot_pred[-1], pred_next],
                    'r--', linewidth=1.5, alpha=0.9)
            ax.plot(next_x, pred_next, 'ro', markersize=10, zorder=5, label='下一交易日预测')
            ax.annotate(f'{pred_next:.2f}', xy=(next_x, pred_next),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=9, color='red', fontweight='bold')

            last_date_str = plot_dates[-1]
            try:
                try:
                    last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
                except ValueError:
                    last_date = datetime.strptime(last_date_str, '%Y%m%d')
                next_label = get_next_trade_date_label(last_date)
            except Exception:
                next_label = '下一交易日'
            x_labels = list(plot_dates) + [next_label]

            step = max(1, (len(x_labels) - 1) // 10)
            tick_positions = list(range(0, len(x_labels) - 1, step))
            if len(x_labels) - 2 not in tick_positions:
                tick_positions.append(len(x_labels) - 2)
            tick_positions.append(len(x_labels) - 1)
            tick_positions = sorted(set(tick_positions))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([x_labels[i] for i in tick_positions], rotation=45, fontsize=8)

            # 指标文本框
            textstr = '\n'.join([
                f"RMSE: {metrics['RMSE']:.4f}",
                f"MAE: {metrics['MAE']:.4f}",
                f"MAPE: {metrics['MAPE']:.2f}%",
                f"年化收益率: {metrics['annualized_return']*100:.2f}%",
                f"夏普比率: {metrics['sharpe_ratio']:.4f}",
            ])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)

            all_prices = np.concatenate([plot_real, plot_pred, [pred_next]])
            y_min = np.min(all_prices)
            y_max = np.max(all_prices)
            y_range = y_max - y_min if y_max > y_min else y_max * 0.1
            ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.55)

            ax.set_title(f"{stock_code} {stock_name} 实际 vs 预测走势", fontsize=14)
            ax.set_xlabel('交易日', fontsize=10)
            ax.set_ylabel('收盘价 (CNY)', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    图表已保存: {fig_path}")
        else:
            print(f"    无回测数据，跳过可视化")

        direction = "涨" if pct_change >= 0 else "跌"

        # 汇总结果
        result = {
            'code': stock_code,
            'name': stock_name,
            'last_close': last_close,
            'pred_close': pred_next,
            'change': change,
            'pct_change': pct_change,
            'has_text': has_text,
            'backtest_samples': len(backtest_real),
            'rmse': metrics['RMSE'],
            'mae': metrics['MAE'],
            'mape': metrics['MAPE'],
            'annualized_return': metrics['annualized_return'],
            'sharpe_ratio': metrics['sharpe_ratio']
        }

        print(f"\n  预测结果: 最新收盘 {last_close:.2f} → 下一交易日预测 {pred_next:.2f} ({direction}{abs(pct_change):.2f}%)")

        return result

    except Exception as e:
        print(f"  预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def cleanup_predict_files():
    """预测后清理中间数据（sentiment_results.csv 和 embeddings .npy 文件）。"""
    print(f"\n{'='*60}")
    print(f"  【预测后清理中间文件】")
    print(f"{'='*60}")

    deleted_csv_count = 0
    csv_pattern = os.path.join(PREDICT_SENTIMENT_DIR, "*_sentiment_results.csv")
    for csv_path in glob.glob(csv_pattern):
        try:
            os.remove(csv_path)
            deleted_csv_count += 1
        except Exception as e:
            print(f"  删除 {csv_path} 失败：{e}")
    print(f"  已删除 {deleted_csv_count} 个 sentiment_results.csv")

    # 2. 删除 embeddings 下的 .npy 文件
    deleted_files = 0
    if os.path.exists(PREDICT_EMBEDDINGS_DIR):
        for sub_dir in os.listdir(PREDICT_EMBEDDINGS_DIR):
            emb_dir = os.path.join(PREDICT_EMBEDDINGS_DIR, sub_dir)
            if os.path.isdir(emb_dir):
                npy_files = glob.glob(os.path.join(emb_dir, "*.npy"))
                for f in npy_files:
                    try:
                        os.remove(f)
                        deleted_files += 1
                    except Exception as e:
                        print(f"  删除 {f} 失败：{e}")
    print(f"  已删除 {deleted_files} 个 embeddings .npy 文件（保留空目录结构）")
    print()


def main_interactive():
    """交互式主程序：用户选择股票并执行预测"""
    print("=" * 60)
    print("  Fin-Transformer 股价预测系统")
    print("=" * 60)
    print(f"  预测日期: 下一交易日")
    print(f"  模型目录: {config.MODELS_DIR}")
    print(f"  价格数据: {PRICE_TEST_DIR}（取最后90天日历天）")
    print(f"  文本数据库: {DB_NAME_TEST}")
    print(f"  输出目录: {PREDICT_OUTPUT_DIR}")
    print("=" * 60)

    # 显示可选股票列表
    print(f"\n可选股票列表：")
    for i, s in enumerate(config.STOCK_LIST, 1):
        print(f"  {i:2d}. {s['code']} - {s['name']}")

    # 交互式选择股票
    print()
    while True:
        try:
            user_input = input(
                f"请输入要预测的股票序号（1-{len(config.STOCK_LIST)}，多只用逗号分隔，输入 all 或直接回车预测全部）："
            ).strip()

            if user_input == "" or user_input.lower() == "all":
                selected_stocks = list(config.STOCK_LIST)
                break

            # 解析逗号分隔的序号
            parts = [p.strip() for p in user_input.split(",") if p.strip()]
            indices = []
            valid = True
            for p in parts:
                if not p.isdigit():
                    print(f"  输入无效：'{p}' 不是合法数字，请重新输入")
                    valid = False
                    break
                idx = int(p)
                if idx < 1 or idx > len(config.STOCK_LIST):
                    print(f"  序号 {idx} 超出范围（1-{len(config.STOCK_LIST)}），请重新输入")
                    valid = False
                    break
                indices.append(idx)

            if not valid:
                continue

            if not indices:
                print("  未输入任何序号，请重新输入")
                continue

            # 去重并保持输入顺序
            seen = set()
            unique_indices = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)

            selected_stocks = [config.STOCK_LIST[idx - 1] for idx in unique_indices]
            break

        except (ValueError, EOFError):
            print("  请输入有效的内容")

    # 显示选中的股票
    print(f"\n将预测以下 {len(selected_stocks)} 只股票：")
    for s in selected_stocks:
        print(f"  - {s['code']} {s['name']}")
    print()

    # 确保输出目录存在
    os.makedirs(PREDICT_SENTIMENT_DIR, exist_ok=True)
    os.makedirs(PREDICT_EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(PREDICT_FIGURES_DIR, exist_ok=True)

    results = []

    for i, stock_info in enumerate(selected_stocks, 1):
        print(f"\n[{i}/{len(selected_stocks)}]", end="")
        result = predict_stock(stock_info)
        if result:
            results.append(result)

    # 打印汇总表
    print(f"\n\n{'='*80}")
    print(f"  预测结果汇总")
    print(f"{'='*80}")
    print(f"{'代码':<8} {'名称':<8} {'最新收盘':>8} {'预测收盘':>8} {'涨跌幅':>8} {'文本':>4} {'RMSE':>8} {'MAPE':>8} {'年化收益率':>10} {'夏普比率':>8}")
    print("-" * 100)

    for r in results:
        direction = "↑" if r['pct_change'] >= 0 else "↓"
        text_status = "有" if r['has_text'] else "无"
        print(f"{r['code']:<8} {r['name']:<8} {r['last_close']:>8.2f} {r['pred_close']:>8.2f} "
              f"{direction}{abs(r['pct_change']):>6.2f}% {text_status:>4} {r['rmse']:>8.4f} {r['mape']:>7.2f}% "
              f"{r['annualized_return']*100:>9.2f}% {r['sharpe_ratio']:>8.4f}")

    print("-" * 100)
    print(f"  共预测 {len(results)}/{len(selected_stocks)} 只股票")

    # 保存CSV汇总
    if results:
        df_results = pd.DataFrame(results)
        csv_path = os.path.join(PREDICT_OUTPUT_DIR, "prediction_summary.csv")
        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  汇总已保存: {csv_path}")

    # 清理预测过程中产生的中间文件
    cleanup_predict_files()

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main_interactive()
