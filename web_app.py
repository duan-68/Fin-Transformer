#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fin-Transformer 股价预测系统 — Streamlit Web 应用
"""

import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 确保项目根目录在 sys.path 中
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pymongo
from datetime import datetime, timedelta
import config
from data.get_data.get_price import fetch_stock_price
from data.get_data.get_post import post_thread
from data.get_data.process import delete_outside_range, fill_missing_dates
from predict_next import (
    rebuild_scaler,
    prepare_price_sequence,
    prepare_text_sequence,
    cleanup_predict_files,
    get_next_trade_date_label,
    PREDICT_FIGURES_DIR,
    PREDICT_SENTIMENT_DIR,
    PREDICT_EMBEDDINGS_DIR,
    PREDICT_OUTPUT_DIR,
)
from src.model import FinTransformerModel, SimpleTransformerModel
from src.evaluate import explain_with_shap
from src.visualize import plot_shap_summary, plot_shap_beeswarm_violin

# ======================== 股票名称映射 ========================
STOCK_INFO = {s['code']: s for s in config.STOCK_LIST}


def get_latest_trade_date():
    """获取最近的交易日作为基准日期"""
    try:
        from data.get_data.get_price import _get_pro
        pro = _get_pro()
        now = datetime.now()
        today_str = now.strftime("%Y%m%d")

        # 查询近30天的交易日历（足够覆盖长假）
        start_cal = (now - timedelta(days=30)).strftime("%Y%m%d")
        df_cal = pro.trade_cal(exchange='SSE', start_date=start_cal, end_date=today_str)
        # df_cal 包含 cal_date(str YYYYMMDD), is_open(int 0/1)

        trade_days = df_cal[df_cal['is_open'] == 1]['cal_date'].sort_values().tolist()

        if not trade_days:
            # 兜底：返回昨天
            return (now - timedelta(days=1)).date()

        today_is_trade_day = today_str in trade_days

        if now.hour >= 15 and today_is_trade_day:
            # 15:00后且今天是交易日 → 基准日期为今天
            return now.date()
        else:
            # 否则 → 最近的上一个交易日
            past_days = [d for d in trade_days if d < today_str]
            if not past_days:
                return (now - timedelta(days=1)).date()
            return datetime.strptime(past_days[-1], "%Y%m%d").date()
    except Exception:
        # tushare API 调用失败，回退到原有逻辑
        now = datetime.now()
        if now.hour >= 15:
            return now.date()
        else:
            return (now - timedelta(days=1)).date()


def get_last_fetch_info():
    """读取上次数据获取的信息"""
    info_path = os.path.join(config.ROOT_DIR, 'last_fetch_info.json')
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def save_fetch_info(ref_date, text_end_str=''):
    """保存本次数据获取的信息"""
    info_path = os.path.join(config.ROOT_DIR, 'last_fetch_info.json')
    info = {
        'ref_date': ref_date.strftime('%Y-%m-%d'),
        'text_end_date': text_end_str,
        'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(info_path, 'w') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def run_data_fetch(force=False, target_stocks=None):
    """获取股票的最新数据。target_stocks 为 None 时获取全部20只。"""

    ref_date = get_latest_trade_date()

    start_date = (ref_date - timedelta(days=90))
    end_date = ref_date

    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    # 文本数据范围（扩展到当前日期，以包含周末帖子）
    today = datetime.now().date()
    if today > ref_date:
        # 当前日期在最近交易日之后（如周末），文本数据范围延伸到今天
        text_end_str = today.strftime("%Y%m%d")
        text_end_datetime = text_end_str + "2359"
    else:
        # 当前日期等于交易日（如交易日收盘后），保持原范围
        text_end_str = end_str
        text_end_datetime = end_str + "2359"

    start_datetime = start_str + "0000"
    end_datetime = end_str + "2359"  # 价格相关的保持不变

    # ---- 检查是否已获取该基准日期的数据（仅对全部股票获取生效）----
    if target_stocks is None:
        last_info = get_last_fetch_info()
        if not force and last_info and last_info.get('ref_date') == ref_date.strftime('%Y-%m-%d') and last_info.get('text_end_date', '') == text_end_str:
            st.success(f"✅ 数据已获取（基准交易日: {ref_date}，获取时间: {last_info.get('fetch_time', '未知')}），无需重复获取。")
            return

    if target_stocks is not None:
        stocks = target_stocks
    else:
        stocks = config.STOCK_LIST

    if target_stocks is not None:
        stock_names = ", ".join([s['code'] for s in stocks])
        if today > ref_date:
            st.info(f"目标股票: {stock_names} | 基准交易日: {ref_date} | 价格数据: {start_date} ~ {end_date} | 文本数据: {start_date} ~ {today}（含周末帖子）")
        else:
            st.info(f"目标股票: {stock_names} | 基准日期: {ref_date} | 数据范围: {start_date} ~ {end_date}")
    elif today > ref_date:
        st.info(f"基准交易日: {ref_date} | 价格数据: {start_date} ~ {end_date} | 文本数据: {start_date} ~ {today}（含周末帖子）")
    else:
        st.info(f"基准日期: {ref_date} | 数据范围: {start_date} ~ {end_date}")

    # ---- 步骤清单 ----
    steps = ["获取价格数据", "获取文本数据", "清洗文本数据"]
    step_placeholders = []
    for i, desc in enumerate(steps):
        ph = st.empty()
        ph.markdown(f"⏳ **[{i+1}/3]** {desc}")
        step_placeholders.append(ph)

    sub_text = st.empty()      # 子进度文本（上方描述）
    sub_progress = st.empty()  # 子进度条

    total = len(stocks)

    # ===== Step 1: 获取价格数据 =====
    prices_dir = os.path.join(config.ROOT_DIR, 'data', 'data', 'prices_test')
    for i, s in enumerate(stocks):
        sub_text.markdown(f"正在获取 **{s['code']}** 价格数据")
        sub_progress.progress((i + 1) / total, text=f"{int((i+1)/total*100)}%")
        fetch_stock_price(s['code'], s['suffix'], start_str, end_str, prices_dir)
        time.sleep(0.5)  # tushare API 限流
    sub_progress.empty()
    sub_text.empty()
    step_placeholders[0].markdown(f"✅ **[1/3]** {steps[0]}")

    # ===== Step 2: 获取文本数据（并发爬取）=====
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_single_stock_posts(code):
        """单只股票文本数据获取（线程内执行）"""
        try:
            client_local = pymongo.MongoClient(config.MONGO_HOST, config.MONGO_PORT)
            local_db = client_local["posts_test"]
            collection = local_db[f"post_{code}"]

            earliest_doc = collection.find_one(sort=[('post_date', 1)])
            latest_doc = collection.find_one(sort=[('post_date', -1)])
            if earliest_doc and latest_doc:
                earliest = earliest_doc['post_date']
                latest = latest_doc['post_date']
                if earliest <= start_str and latest >= text_end_str:
                    client_local.close()
                    return code, 'skipped', None

            client_local.close()
            post_thread(code, 1, 300, cutoff_date=start_str, target_start_date=start_str)
            return code, 'done', None
        except Exception as e:
            return code, 'error', str(e)

    skipped_count = 0
    error_codes = []

    sub_text.markdown(f"正在并发爬取文本数据（最大并发: {config.CRAWLER_MAX_WORKERS}）...")

    with ThreadPoolExecutor(max_workers=config.CRAWLER_MAX_WORKERS) as executor:
        futures = {}
        for s in stocks:
            future = executor.submit(_fetch_single_stock_posts, s['code'])
            futures[future] = s['code']

        completed_count = 0
        for future in as_completed(futures):
            code = futures[future]
            status, result_status, error_msg = future.result()
            completed_count += 1

            sub_progress.progress(completed_count / total, text=f"{int(completed_count/total*100)}%")

            if result_status == 'skipped':
                skipped_count += 1
                sub_text.markdown(f"**{code}** 数据已覆盖，跳过 ({completed_count}/{total})")
            elif result_status == 'error':
                error_codes.append(code)
                st.warning(f"{code} 文本获取失败: {error_msg}")
                sub_text.markdown(f"**{code}** 获取失败 ({completed_count}/{total})")
            else:
                sub_text.markdown(f"**{code}** 爬取完成 ({completed_count}/{total})")

    sub_progress.empty()
    sub_text.empty()
    skip_msg = f"（其中 {skipped_count} 只已有数据，跳过爬取）" if skipped_count > 0 else ""
    error_msg_text = f"（{len(error_codes)} 只获取失败）" if error_codes else ""
    step_placeholders[1].markdown(f"✅ **[2/3]** {steps[1]} {skip_msg}{error_msg_text}")

    # ===== Step 3: 清洗文本数据 =====
    client = pymongo.MongoClient(config.MONGO_HOST, config.MONGO_PORT)
    db = client["posts_test"]
    for i, s in enumerate(stocks):
        sub_text.markdown(f"正在清洗 **{s['code']}** 文本数据")
        sub_progress.progress((i + 1) / total, text=f"{int((i+1)/total*100)}%")
        collection = db[f"post_{s['code']}"]
        try:
            delete_outside_range(collection, start_datetime, text_end_datetime)
            fill_missing_dates(collection, s['code'], start_str, text_end_str)
        except Exception as e:
            st.warning(f"{s['code']} 清洗失败: {e}")
    client.close()
    sub_progress.empty()
    sub_text.empty()
    step_placeholders[2].markdown(f"✅ **[3/3]** {steps[2]}")

    # 仅全部获取时更新缓存
    if target_stocks is None:
        save_fetch_info(ref_date, text_end_str)
    st.success("数据获取完成！")


def run_prediction(stock_code: str, model_type: str = 'fin-transformer'):
    """逐步执行预测流程，返回预测结果字典，失败返回 None。"""
    stock_name = STOCK_INFO[stock_code]['name']

    st.subheader("预测进度")
    if model_type == 'fin-transformer':
        model_display = f"Fin-Transformer_{stock_code}"
    else:
        model_display = f"Transformer_{stock_code}"
    st.markdown(f"**正在预测：{stock_name}（{stock_code}）| 模型：{model_display}**")

    steps = [
        "重建训练数据 Scaler",
        "准备价格序列",
        "准备文本序列（FinBERT 推理）",
        "加载模型并执行回测推理",
        "计算回测指标",
        "生成可视化图表",
    ]

    step_placeholders = []
    for i, desc in enumerate(steps):
        ph = st.empty()
        ph.markdown(f"⏳ **[{i+1}/6]** {desc}")
        step_placeholders.append(ph)

    finbert_progress_text = st.empty()
    finbert_progress_bar = st.empty()

    def update_step(step_idx, done=False):
        desc = steps[step_idx]
        if done:
            step_placeholders[step_idx].markdown(f"✅ **[{step_idx+1}/6]** {desc}")
        else:
            step_placeholders[step_idx].markdown(f"⏳ **[{step_idx+1}/6]** {desc}...")

    def finbert_callback(current, total):
        pct = current / total if total > 0 else 0
        finbert_progress_text.markdown(f"FinBERT 情感分析推理中（{current}/{total}）")
        finbert_progress_bar.progress(pct, text=f"{int(pct * 100)}%")

    try:
        # 确保输出目录存在
        os.makedirs(PREDICT_SENTIMENT_DIR, exist_ok=True)
        os.makedirs(PREDICT_EMBEDDINGS_DIR, exist_ok=True)
        os.makedirs(PREDICT_FIGURES_DIR, exist_ok=True)

        # 检查前置文件
        model_path = config.get_model_save_path(stock_code, model_type=model_type)
        if not os.path.exists(model_path):
            st.error(f"模型文件不存在: {model_path}")
            return None

        price_test_dir = os.path.join(ROOT_DIR, "data", "data", "prices_test")
        price_path = os.path.join(price_test_dir, f"{stock_code}.csv")
        if not os.path.exists(price_path):
            st.error(f"价格数据不存在: {price_path}")
            return None

        update_step(0)
        scaler, scaler_info = rebuild_scaler(stock_code)
        update_step(0, done=True)

        update_step(1)
        all_norm, all_dates, all_close = prepare_price_sequence(stock_code, scaler)
        update_step(1, done=True)

        update_step(2)
        if model_type == 'transformer':
            text_vecs = np.zeros((len(all_dates), 768), dtype=np.float32)
            has_text = False
            # 跳过 FinBERT 进度条
        else:
            text_vecs, has_text = prepare_text_sequence(
                stock_code, all_dates, None,
                progress_callback=finbert_callback
            )
            # 清理 FinBERT 进度显示
            finbert_progress_text.empty()
            finbert_progress_bar.empty()
        update_step(2, done=True)

        update_step(3)

        device = torch.device(config.DEVICE)

        if model_type == 'transformer':
            model = SimpleTransformerModel(
                num_input_dim=len(config.FEATURE_COLS),
                d_model=128,
                nhead=4,
                num_layers=2,
                dim_feedforward=256,
                dropout=0.1,
                pred_length=config.PRED_LENGTH,
            ).to(device)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
        else:
            model = FinTransformerModel(
                num_input_dim=len(config.FEATURE_COLS),
                text_input_dim=768,
                d_model=config.D_MODEL,
                nhead=config.NHEAD,
                num_layers=config.NUM_LAYERS,
                dim_feedforward=config.DIM_FEEDFORWARD,
                dropout=config.DROPOUT,
                pred_length=config.PRED_LENGTH,
            ).to(device)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()

        seq_len = config.SEQ_LENGTH
        n_total = len(all_norm)

        backtest_real = []
        backtest_pred = []
        backtest_dates = []

        for i in range(n_total - seq_len):
            x_num_np = all_norm[i : i + seq_len]
            x_text_np = text_vecs[i : i + seq_len]

            x_num_t = torch.tensor(x_num_np, dtype=torch.float32).unsqueeze(0).to(device)
            x_text_t = torch.tensor(x_text_np, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                if model_type == 'transformer':
                    pred_norm = model(x_num_t)
                else:
                    pred_norm, _ = model(x_num_t, x_text_t)

            pred_price = pred_norm.item() * scaler_info['close_range'] + scaler_info['close_min']
            real_price = all_close[i + seq_len]

            backtest_pred.append(pred_price)
            backtest_real.append(real_price)
            backtest_dates.append(all_dates[i + seq_len])

        # 下一交易日预测
        x_num_last = all_norm[-seq_len:]
        x_text_last = text_vecs[-seq_len:]
        x_num_t = torch.tensor(x_num_last, dtype=torch.float32).unsqueeze(0).to(device)
        x_text_t = torch.tensor(x_text_last, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            if model_type == 'transformer':
                pred_norm_next = model(x_num_t)
            else:
                pred_norm_next, _ = model(x_num_t, x_text_t)

        pred_next = pred_norm_next.item() * scaler_info['close_range'] + scaler_info['close_min']
        last_close = float(all_close[-1])
        change = pred_next - last_close
        pct_change = change / last_close * 100

        # 诊断日志
        if model_type == 'fin-transformer':
            print(f"[诊断] Fin-Transformer 原始预测值: {pred_norm_next.item():.6f}")
            print(f"[诊断] 文本向量非零天数: {np.count_nonzero(np.any(text_vecs != 0, axis=1))}/{len(text_vecs)}")
        else:
            print(f"[诊断] SimpleTransformer 原始预测值: {pred_norm_next.item():.6f}")

        update_step(3, done=True)
        update_step(4)

        backtest_real = np.array(backtest_real)
        backtest_pred = np.array(backtest_pred)

        if len(backtest_real) > 0:
            rmse = float(np.sqrt(np.mean((backtest_real - backtest_pred) ** 2)))
            mae = float(np.mean(np.abs(backtest_real - backtest_pred)))
            mape = float(np.mean(np.abs((backtest_real - backtest_pred) / backtest_real)) * 100)

            n_days = len(backtest_real)
            if n_days > 1 and backtest_real[0] != 0:
                total_return = backtest_real[-1] / backtest_real[0]
                annualized_return = float(total_return ** (252 / n_days) - 1)
            else:
                annualized_return = 0.0

            if len(backtest_pred) > 1:
                daily_returns = np.diff(backtest_pred) / backtest_pred[:-1]
                std_returns = np.std(daily_returns)
                if std_returns > 0:
                    sharpe_ratio = float(np.mean(daily_returns) / std_returns * np.sqrt(252))
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0

            metrics = {
                'RMSE': rmse, 'MAE': mae, 'MAPE': mape,
                'annualized_return': annualized_return, 'sharpe_ratio': sharpe_ratio,
            }
        else:
            metrics = {
                'RMSE': 0, 'MAE': 0, 'MAPE': 0,
                'annualized_return': 0.0, 'sharpe_ratio': 0.0,
            }

        update_step(4, done=True)
        update_step(5)

        fig_path = os.path.join(PREDICT_FIGURES_DIR, f"predict_{stock_code}.png")

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

            model_label = 'Fin-Transformer' if model_type == 'fin-transformer' else 'Transformer'
            ax.set_title(f"{stock_code} {stock_name} 实际 vs {model_label} 预测走势", fontsize=14)
            ax.set_xlabel('交易日', fontsize=10)
            ax.set_ylabel('收盘价 (CNY)', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        update_step(5, done=True)

        X_num_samples = []
        X_text_samples = []
        for i in range(n_total - seq_len):
            X_num_samples.append(all_norm[i : i + seq_len])
            X_text_samples.append(text_vecs[i : i + seq_len])
        X_num_sample = np.array(X_num_samples) if X_num_samples else None
        if model_type == 'transformer':
            X_text_sample = None
        else:
            X_text_sample = np.array(X_text_samples) if X_text_samples else None

        result = {
            'code': stock_code,
            'name': stock_name,
            'last_close': last_close,
            'pred_close': pred_next,
            'change': change,
            'pct_change': pct_change,
            'has_text': has_text,
            'metrics': metrics,
            'fig_path': fig_path,
            'model': model,
            'device': device,
            'X_num_sample': X_num_sample,
            'X_text_sample': X_text_sample,
            'model_type': model_type,
            'backtest_real': backtest_real,
            'backtest_pred': backtest_pred,
            'backtest_dates': backtest_dates,
        }

        # --- Fin-Transformer 选择时，额外运行 Transformer 基线对比 ---
        if model_type == 'fin-transformer':
            transformer_model_path = config.get_model_save_path(stock_code, model_type='transformer')
            if os.path.exists(transformer_model_path):
                simple_model = SimpleTransformerModel(
                    num_input_dim=len(config.FEATURE_COLS),
                    d_model=128, nhead=4, num_layers=2,
                    dim_feedforward=256, dropout=0.1,
                    pred_length=config.PRED_LENGTH,
                ).to(device)
                simple_state = torch.load(transformer_model_path, map_location=device)
                simple_model.load_state_dict(simple_state)
                simple_model.eval()

                # 用同样的数值数据做回测
                simple_backtest_pred = []
                with torch.no_grad():
                    for i in range(len(all_norm) - seq_len):
                        x_num_t = torch.FloatTensor(all_norm[i:i+seq_len]).unsqueeze(0).to(device)
                        p = simple_model(x_num_t)
                        val = p.item() * scaler_info['close_range'] + scaler_info['close_min']
                        simple_backtest_pred.append(val)

                # 下一交易日预测
                x_last = torch.FloatTensor(all_norm[-seq_len:]).unsqueeze(0).to(device)
                simple_pred_next_norm = simple_model(x_last)
                simple_pred_next = simple_pred_next_norm.item() * scaler_info['close_range'] + scaler_info['close_min']

                print(f"[诊断] SimpleTransformer 原始预测值: {simple_pred_next_norm.item():.6f}")
                print(f"[诊断] 双模型预测差异: Fin={pred_next:.4f}, Simple={simple_pred_next:.4f}, 差值={pred_next - simple_pred_next:.4f}")

                result['simple_backtest_pred'] = simple_backtest_pred
                result['simple_pred_next'] = simple_pred_next

        return result

    except Exception as e:
        import traceback
        st.error(f"预测失败: {e}\n{traceback.format_exc()}")
        return None


def main():
    st.set_page_config(page_title="Fin-Transformer 股价预测系统", layout="wide")
    st.title("欢迎使用 Fin-Transformer 股价预测系统！")
    st.text("姓名：江昊择"+"     学号：2022080301010")

    st.sidebar.header("数据管理")

    fetch_scope = st.sidebar.radio(
        "获取范围",
        ["全部股票", "单只股票"],
        index=0,
        horizontal=True
    )

    if fetch_scope == "单只股票":
        fetch_stock_options = [f"{s['code']} - {s['name']}" for s in config.STOCK_LIST]
        fetch_selected = st.sidebar.selectbox("选择目标股票", fetch_stock_options, key="fetch_stock_select")
        fetch_stock_code = fetch_selected.split(" - ")[0]
    else:
        fetch_stock_code = None

    fetch_btn = st.sidebar.button("📥 获取最新数据", use_container_width=True)
    force_fetch = st.sidebar.checkbox("强制重新获取（忽略缓存）", value=False)
    st.sidebar.markdown("---")
    st.sidebar.header("参数设置")
    stock_options = [f"{s['code']} - {s['name']}" for s in config.STOCK_LIST]
    selected = st.sidebar.selectbox("选择股票", stock_options)
    stock_code = selected.split(" - ")[0]

    model_type = st.sidebar.radio(
        "选择模型",
        ["Fin-Transformer (双流融合)", "Transformer (纯数值基线)"],
        index=0
    )
    # 解析模型类型
    use_model_type = 'fin-transformer' if 'Fin-Transformer' in model_type else 'transformer'

    predict_btn = st.sidebar.button("🚀 开始预测", use_container_width=True)

    st.sidebar.markdown("---")
    if st.sidebar.button("🛑 退出应用", use_container_width=True):
        import streamlit.components.v1 as components
        # 用 JS 替换整个页面内容为关闭提示，然后终止服务
        components.html("""
            <script>
                // 替换 Streamlit 父页面内容
                window.parent.document.body.innerHTML = '\
                    <div style="display:flex;flex-direction:column;justify-content:center;align-items:center;height:100vh;background:#0e1117;color:#fafafa;font-family:sans-serif;">\
                        <h2 style="margin-bottom:12px;">应用已关闭</h2>\
                        <p style="color:#888;">请手动关闭此标签页</p>\
                    </div>';
            </script>
        """, height=0)
        time.sleep(1)
        print("应用已关闭")
        os._exit(0)

    # ---- 主区域 ----
    if fetch_btn:
        if fetch_stock_code:
            # 单只股票：从 STOCK_LIST 中筛选
            target = [s for s in config.STOCK_LIST if s['code'] == fetch_stock_code]
            run_data_fetch(force=force_fetch, target_stocks=target)
        else:
            # 全部股票
            run_data_fetch(force=force_fetch)

    if predict_btn:
        result = run_prediction(stock_code, model_type=use_model_type)

        if result is None:
            return

        st.markdown("---")

        st.subheader("📈 下一交易日预测结果")

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("下一交易日预测收盘价", f"¥{result['pred_close']:.2f}")
        col_m2.metric("预测涨跌幅", f"{result['pct_change']:+.2f}%",
                       delta=f"{result['pct_change']:+.2f}%")
        col_m3.metric("预测涨跌额", f"¥{result['change']:+.2f}",
                       delta=f"{result['change']:+.2f}")

        text_status = "✅ 使用文本（舆情）数据" if result['has_text'] else "⚠️ 无文本数据，使用零向量替代"
        st.info(text_status)

        st.markdown("---")

        st.subheader("📊 走势对比图")
        if os.path.exists(result['fig_path']):
            st.image(result['fig_path'], use_container_width=True)
        else:
            st.warning("走势图未生成（回测数据不足）")

        # 双模型预测对比（仅 Fin-Transformer 选择时）
        if result.get('simple_backtest_pred') is not None:
            st.subheader("📊 双模型预测对比")

            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            fig_cmp, ax_cmp = plt.subplots(figsize=(14, 6))

            bt_real = result['backtest_real']
            bt_pred_ft = result['backtest_pred']
            bt_pred_simple = np.array(result['simple_backtest_pred'])
            bt_dates = result['backtest_dates']

            n_show = min(7, len(bt_real))
            plot_real = bt_real[-n_show:]
            plot_pred_ft = bt_pred_ft[-n_show:]
            plot_pred_simple = bt_pred_simple[-n_show:]
            plot_dates = bt_dates[-n_show:]

            x_idx = range(n_show)
            ft_pred_next = result['pred_close']
            simple_pred_next = result['simple_pred_next']

            # 实际值
            ax_cmp.plot(x_idx, plot_real, 'b-o', linewidth=1.5, label='实际收盘价', alpha=0.9)

            # Fin-Transformer 预测
            ax_cmp.plot(x_idx, plot_pred_ft, 'r--s', linewidth=1.5, label='Fin-Transformer 预测', alpha=0.9)
            next_x = n_show
            ax_cmp.plot(next_x, ft_pred_next, 'rs', markersize=10, zorder=5, label='Fin-Transformer 下一交易日预测')
            ax_cmp.plot([n_show - 1, next_x], [plot_pred_ft[-1], ft_pred_next], 'r--', linewidth=1.5, alpha=0.5)

            # Transformer 预测
            ax_cmp.plot(x_idx, plot_pred_simple, 'g--^', linewidth=1.5, label='Transformer 预测', alpha=0.9)
            ax_cmp.plot(next_x, simple_pred_next, 'g^', markersize=10, zorder=5, label='Transformer 下一交易日预测')
            ax_cmp.plot([n_show - 1, next_x], [plot_pred_simple[-1], simple_pred_next], 'g--', linewidth=1.5, alpha=0.5)

            stock_name = result['name']
            ax_cmp.set_title(f'{stock_name}（{stock_code}）双模型预测对比', fontsize=14)
            ax_cmp.set_xlabel('交易日', fontsize=10)
            ax_cmp.set_ylabel('收盘价 (CNY)', fontsize=10)
            ax_cmp.legend(loc='upper left', fontsize=8)
            ax_cmp.grid(True, alpha=0.3)

            # x轴日期标签
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
            ax_cmp.set_xticks(range(len(x_labels)))
            ax_cmp.set_xticklabels(x_labels, rotation=45, fontsize=8)

            all_prices_cmp = np.concatenate([plot_real, plot_pred_ft, plot_pred_simple,
                                             [ft_pred_next, simple_pred_next]])
            y_min = np.min(all_prices_cmp)
            y_max = np.max(all_prices_cmp)
            y_range = y_max - y_min if y_max > y_min else y_max * 0.1
            ax_cmp.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.15)

            plt.tight_layout()
            cmp_fig_path = os.path.join(PREDICT_FIGURES_DIR, f'compare_{stock_code}.png')
            fig_cmp.savefig(cmp_fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig_cmp)
            st.image(cmp_fig_path, use_container_width=True)

        st.markdown("---")

        st.subheader("📋 回测指标")
        metrics = result['metrics']
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("RMSE", f"{metrics['RMSE']:.4f}")
        c2.metric("MAE", f"{metrics['MAE']:.4f}")
        c3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
        c4.metric("年化收益率", f"{metrics['annualized_return']*100:.2f}%")
        c5.metric("夏普比率", f"{metrics['sharpe_ratio']:.4f}")

        st.markdown("---")

        if result['model_type'] != 'transformer':
            st.subheader("🔍 SHAP 可解释性分析")

            if result['X_num_sample'] is not None and len(result['X_num_sample']) > 0:
                with st.spinner("正在进行SHAP可解释性分析..."):
                    shap_results = explain_with_shap(
                        result['model'],
                        result['X_num_sample'],
                        result['X_text_sample'],
                        result['device'],
                        feature_names_num=config.FEATURE_COLS,
                    )

                    save_dir = PREDICT_FIGURES_DIR

                    # 抑制 plt.show()，保存图表后用 st.image 展示
                    import matplotlib.pyplot as _plt
                    _orig_show = _plt.show
                    _plt.show = lambda *a, **k: None
                    try:
                        plot_shap_summary(shap_results, stock_code, save_dir=save_dir)
                        plot_shap_beeswarm_violin(shap_results, stock_code, save_dir=save_dir)
                    finally:
                        _plt.show = _orig_show
                        _plt.close('all')

                    shap_summary_path = os.path.join(save_dir, f"shap_summary_{stock_code}.png")
                    shap_beeswarm_path = os.path.join(save_dir, f"shap_beeswarm_violin_{stock_code}.png")

                    if os.path.exists(shap_summary_path):
                        st.image(shap_summary_path, caption="SHAP 特征重要性汇总", use_container_width=True)
                    if os.path.exists(shap_beeswarm_path):
                        st.image(shap_beeswarm_path, caption="SHAP 蜂群图 + 小提琴图", use_container_width=True)
            else:
                st.warning("回测样本不足，无法进行 SHAP 分析")

        # 清理临时文件 
        cleanup_predict_files()


main()

# 启动：python -m streamlit run web_app.py --server.headless true