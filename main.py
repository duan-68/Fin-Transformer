"""
主程序入口：数据准备 → 样本构建 → 训练/评估 → 可视化
"""

import os
import argparse
import torch
import numpy as np
import shap
from torch.utils.data import DataLoader

import config
from src.data import (
    load_price_data,
    load_text_vectors,
    create_dual_sequences,
    DualStreamDataset,
    run_finbert_inference,
    aggregate_daily_text_vectors
)
from src.model import FinTransformerModel
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import plot_prediction, plot_training_history, plot_shap_beeswarm_violin
from src.utils import set_seed

from src.evaluate import explain_with_shap   
from src.visualize import plot_shap_summary  

set_seed(config.SEED)

def prepare_data(stock_code):
    """确保文本向量已生成，若不存在则自动运行 FinBERT 推理和聚合。"""
    sentiment_csv = config.get_sentiment_csv_path(stock_code)
    if not os.path.exists(sentiment_csv):
        print(f"情感分析结果不存在，开始 FinBERT 推理...")
        run_finbert_inference(stock_code)

    vec_path = config.get_text_vec_path(stock_code)
    if not os.path.exists(vec_path):
        print(f"每日文本向量不存在，开始聚合...")
        price_df = load_price_data(stock_code=stock_code)
        aggregate_daily_text_vectors(stock_code, price_df)
    else:
        print("文本向量已存在，跳过生成步骤。")

def main(args):
    stock_code = config.STOCK_CODE
    target_stock_id = stock_code + config.STOCK_SUFFIX

    print("=" * 60)
    print(f"股票预测流程启动: {stock_code} (数据中 ID: {target_stock_id})")
    print("=" * 60)

    prepare_data(stock_code)
    df_price = load_price_data(stock_code=stock_code)

    daily_vecs, text_days = load_text_vectors(stock_code)
    if daily_vecs is None:
        print(f"无法为股票 {stock_code} 生成有效文本向量，程序终止。")
        return

    (X_train_num, X_train_text, y_train, ids_train,
     X_val_num, X_val_text, y_val, ids_val,
     X_test_num, X_test_text, y_test, ids_test,
     scalers) = create_dual_sequences(
        df_price, daily_vecs, text_days,
        config.FEATURE_COLS, config.TARGET_COL,
        config.SEQ_LENGTH, config.PRED_LENGTH,
        config.TRAIN_RATIO, config.VAL_RATIO
    )

    if len(X_train_num) == 0:
        print("错误：训练集为空，请检查数据。")
        return

    train_dataset = DualStreamDataset(X_train_num, X_train_text, y_train, ids_train)
    val_dataset = DualStreamDataset(X_val_num, X_val_text, y_val, ids_val)
    test_dataset = DualStreamDataset(X_test_num, X_test_text, y_test, ids_test)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = FinTransformerModel(
        num_input_dim=len(config.FEATURE_COLS),
        text_input_dim=768,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        pred_length=config.PRED_LENGTH
    ).to(config.DEVICE)

    model_save_path = config.get_model_save_path(stock_code)

    if args.mode == 'train':
        print("\n开始训练...")
        history = train_model(
            model, train_loader, val_loader,
            epochs=config.EPOCHS,
            lr=config.LR,
            device=config.DEVICE,
            save_path=model_save_path,
            verbose=True
        )
        plot_training_history(history, save_path=os.path.join(config.FIGURES_DIR, f'training_history_{stock_code}.png'))
    else:
        if not os.path.exists(model_save_path):
            print(f"模型文件 {model_save_path} 不存在，请先训练。")
            return
        model.load_state_dict(torch.load(model_save_path, map_location=config.DEVICE))
        print(f"已加载模型: {model_save_path}")

    print("\n测试集评估...")
    scaler_info = scalers[target_stock_id]
    real_price, pred_price, metrics = evaluate_model(
        model, test_loader, scaler_info, target_stock_id, config.DEVICE
    )
    print(f"评估结果: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%")
    print(f"          年化收益率={metrics['annualized_return'] * 100:.2f}%, 夏普比率={metrics['sharpe_ratio']:.4f}")

    figure_path = config.get_figure_save_path(stock_code)
    plot_prediction(real_price, pred_price, stock_code, metrics, save_path=figure_path)
    
    # SHAP 可解释性分析
    print("\n生成 SHAP 可解释性报告...")
    n_explain = min(200, len(X_test_num))
    X_num_explain = X_test_num[:n_explain]
    X_text_explain = X_test_text[:n_explain]

    shap_results = explain_with_shap(
        model,
        X_num_explain,
        X_text_explain,
        device=config.DEVICE,
        feature_names_num=config.FEATURE_COLS,
        background_size=50
    )
    
    print("数值特征原始重要性:", shap_results['num_feature_importance'])
    print("文本时间步重要性:", shap_results['text_temporal_importance_mean'])

    np.savez(os.path.join(config.REPORTS_DIR, f'shap_{stock_code}.npz'),
                num_importance=shap_results['num_feature_importance'],
                text_temporal_mean=shap_results['text_temporal_importance_mean'])

    # 可视化并保存图表
    plot_shap_summary(shap_results, stock_code,
                        save_dir=config.FIGURES_DIR)
    plot_shap_beeswarm_violin(shap_results, stock_code, save_dir=config.FIGURES_DIR)
    print(f"SHAP 解释图表已保存至 {config.FIGURES_DIR}")


    print("\n流程结束。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    args = parser.parse_args()
    main(args)