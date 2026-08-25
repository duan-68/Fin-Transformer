# Fin-Transformer 股价预测系统

## 1. 项目简介

融合数值时序数据与金融文本情感的双流深度学习模型，用于A股市场20只蓝筹股的收盘价预测与可解释性分析。

系统采用 **Fin-Transformer** 双流融合架构：数值流接收7维股价特征（开高低收、成交量、成交额、涨跌幅），文本流接收768维 FinBERT 嵌入向量（由基于股吧标注数据微调后的 FinBERT 模型提取），通过交叉注意力机制融合两路信息，并引入基于波动率与成交量的金融先验偏置增强注意力分配，最终输出收盘价预测结果。

除了核心的 Fin-Transformer 外，系统还实现了多个基线/对比模型用于消融验证：
- **SimpleTransformer**：纯数值单流 Transformer（消融基线）
- **ConcatFusion**：双流编码 + 拼接融合（消融用）
- **CrossAttnOnly**：双流编码 + 交叉注意力（无先验偏置，消融用）
- **LSTM**：经典长短期记忆网络（对比基线）
- **BiLSTM-Attention**：双向 LSTM + 时间注意力（含文本流，对比基线）
- **Informer**：ProbSparse 注意力 Transformer（对比基线）

模型训练完成后，使用 SHAP GradientExplainer 对特征贡献进行可解释性分析。

## 2. 项目结构

```
project/
├── config.py                          # 中央配置管理
├── main.py                            # 单股训练/评估主程序
├── predict_next.py                    # 下一交易日股价预测
├── web_app.py                         # Streamlit Web 应用
├── script.py                          # 数据维护工具
├── requirements.txt                   # 依赖包
├── last_fetch_info.json               # 数据获取缓存记录（自动生成）
├── README.md
│
├── src/                               # 核心源代码
│   ├── data.py                        # 数据处理（加载/FinBERT推理/文本聚合/样本构建）
│   ├── model.py                       # 模型定义（FinTransformer + 3个消融变体）
│   ├── comparison_models.py           # 对比模型定义（LSTM / BiLSTM-Attention / Informer）
│   ├── train.py                       # 训练流程（注意力正则化 + 动态学习率）
│   ├── evaluate.py                    # 评估指标 + SHAP 解释
│   ├── predict.py                     # 预测与反归一化
│   ├── visualize.py                   # 可视化（预测图/训练曲线/SHAP图）
│   └── utils.py                       # 工具函数（中文字体/随机种子）
│
├── tests/                             # 单元测试（pytest，23个用例，覆盖4种模型变体）
│   ├── test_model.py                  # 模型单元测试
│   └── test_data.py                   # 数据管道单元测试
│
├── experiments/                       # 实验与验证
│   ├── ablation_train.py              # 四级消融实验（20股 × 4模型 = 80组）
│   ├── comparison_train.py            # 五模型对比实验（20股 × 5模型 = 100组）
│   ├── plot_comparison.py             # 对比结果可视化（柱状图/雷达图/热力图）
│   ├── robustness_test.py             # 鲁棒性实验（窗口长度 + 随机种子）
│   ├── statistical_tests.py           # 统计显著性检验（Wilcoxon + t检验）
│   └── results/                       # 实验结果输出
│       ├── ablation_results.csv       # 消融实验指标
│       ├── model_comparison.csv       # 五模型对比指标
│       ├── robustness_window.csv      # 窗口长度敏感性结果
│       ├── robustness_seed.csv        # 随机种子稳定性结果
│       ├── statistical_tests.csv      # Wilcoxon 逐股票检验结果
│       ├── mape_improvement_ttest.json # 整体 MAPE 改善率 t 检验
│       └── predictions/               # 逐日预测结果（npz 文件）
│
├── data/
│   ├── data/
│   │   ├── prices/                    # 2025年训练股价 CSV
│   │   ├── prices_test/               # 2026年预测股价 CSV
│   │   └── posts/                     # 股吧帖子 JSON 备份
│   ├── get_data/
│   │   ├── get_price.py               # Tushare API 获取股价
│   │   ├── get_post.py                # Selenium 爬虫获取股吧评论
│   │   ├── clone.py                   # 爬虫克隆版本（并行加速）
│   │   ├── process.py                 # MongoDB 数据清洗与补全
│   │   └── stealth.min.js             # 爬虫反检测脚本
│   ├── processed/                     # 每日文本向量（20只股票各一个目录）
│   │   └── {stock_code}/
│   │       ├── daily_text_vectors.npy # (N, 768) 每日文本向量
│   │       └── trading_days.npy       # 对应的交易日列表
│   └── top20_stock.csv                # 目标股票列表
│
├── drivers/                           # 本地化 Selenium 浏览器及驱动
│   ├── chrome/                        # Chrome 浏览器
│   └── chromedriver/                  # ChromeDriver 驱动
│
├── models/
│   ├── Fin-Transformer/               # Fin-Transformer 双流融合模型（20只）
│   │   └── Fin-Transformer_{code}.pth
│   ├── Transformer/                   # SimpleTransformer 纯数值基线（20只）
│   │   └── Transformer_{code}.pth
│   ├── ablation/                      # 消融实验模型权重（80个 pth）
│   │   └── {ModelType}_{code}.pth
│   ├── LSTM/                          # LSTM 对比模型
│   ├── BiLSTM-Attention/              # BiLSTM-Attention 对比模型
│   ├── Informer/                      # Informer 对比模型
│   ├── FinBERT/
│   │   ├── finetuned_finbert_guba/    # 微调后的 FinBERT 模型（三分类：消极/中性/积极）
│   │   └── train/                     # 微调代码（train.py）、标注数据集（train.csv）
│   └── loss_experiment/               # 损失函数实验模型
│
├── reports/
│   ├── sentiment_analysis/            # 情感分析结果
│   │   ├── {code}_sentiment_results.csv
│   │   └── embeddings/{code}/         # FinBERT 嵌入向量
│   ├── predict/                       # 预测输出
│   │   ├── figures/                   # 预测对比图
│   │   ├── sentiment/embeddings/      # 预测时使用的文本向量（临时）
│   │   └── prediction_summary.csv     # 预测汇总表
│   ├── figures/                       # 训练结果图表
│   │   ├── training_history_{code}.png
│   │   ├── Fin-Transformer_result_{code}.png
│   │   ├── shap_summary_{code}.png
│   │   └── shap_beeswarm_violin_{code}.png
│   └── shap_*.npz
│
└── picture/                           # 论文插图素材
```

## 3. 模型架构

### Fin-Transformer（双流融合）

**数值流**

```
输入 (B, 50, 7) → Linear投影 (7→128) → PositionalEncoding → TransformerEncoder (2层, 4头)
```

**文本流**

```
输入 (B, 50, 768) → Linear投影 (768→128) → PositionalEncoding → TransformerEncoder (2层, 4头)
```

**交叉注意力融合**

- 数值流输出作为 Query，文本流输出作为 Key/Value
- **金融先验偏置**：从波动率和成交量特征生成注意力偏置矩阵，叠加至注意力分数上，引导模型关注高波动、高成交量时段的文本信息
- 残差连接 + LayerNorm

**预测头**

```
融合特征 (B, 128) → Linear (128→256) → GELU → Dropout → Linear (256→1) → 收盘价预测
```

### 消融模型变体

| 模型 | 数值流 | 文本流 | 交叉注意力 | 金融先验偏置 | 用途 |
|------|--------|--------|------------|-------------|------|
| SimpleTransformer | ✓ | ✗ | ✗ | ✗ | 纯数值基线 |
| ConcatFusion | ✓ | ✓ | ✗ | ✗ | 拼接融合对照 |
| CrossAttnOnly | ✓ | ✓ | ✓ | ✗ | 先验偏置消融 |
| FinTransformer | ✓ | ✓ | ✓ | ✓ | 完整模型 |

### 对比模型

| 模型 | 类型 | 描述 |
|------|------|------|
| LSTM | 纯数值 | 两层 LSTM 编码 + MLP 预测头 |
| BiLSTM-Attention | 双流 | BiLSTM 编码 + 时间注意力聚合 + MLP |
| Informer | 纯数值 | ProbSparse 自注意力 Transformer |

### SimpleTransformer（基线对照）

```
输入 (B, 50, 7) → Linear投影 (7→64) → PositionalEncoding → TransformerEncoder (1层, 2头) → 预测头 (Linear→GELU→Dropout→Linear) → 收盘价预测
```

## 4. 目标股票（20只）

| 序号 | 代码   | 名称     | 市场 |
|------|--------|----------|------|
| 1    | 601288 | 农业银行 | SH   |
| 2    | 601857 | 中国石油 | SH   |
| 3    | 601398 | 工商银行 | SH   |
| 4    | 600519 | 贵州茅台 | SH   |
| 5    | 300750 | 宁德时代 | SZ   |
| 6    | 601988 | 中国银行 | SH   |
| 7    | 601138 | 工业富联 | SH   |
| 8    | 601628 | 中国人寿 | SH   |
| 9    | 600036 | 招商银行 | SH   |
| 10   | 601899 | 紫金矿业 | SH   |
| 11   | 601088 | 中国神华 | SH   |
| 12   | 601318 | 中国平安 | SH   |
| 13   | 600900 | 长江电力 | SH   |
| 14   | 600028 | 中国石化 | SH   |
| 15   | 300308 | 中际旭创 | SZ   |
| 16   | 688041 | 海光信息 | SH   |
| 17   | 000333 | 美的集团 | SZ   |
| 18   | 688256 | 寒武纪   | SH   |
| 19   | 601728 | 中国电信 | SH   |
| 20   | 000858 | 五粮液   | SZ   |

## 5. 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

额外依赖（未列入 requirements.txt 的可选/特殊包）：

```bash
pip install transformers pymongo tushare selenium shap
```

### 启动 Web 应用

```bash
python -m streamlit run web_app.py --server.headless true
```

Web 应用功能：

- **模型选择**：侧边栏切换 Fin-Transformer（双流融合）与 Transformer（纯数值基线）
- **数据获取**：
  - 支持"全部股票"一键获取或"单只股票"定向获取
  - 通过 `last_fetch_info.json` 记录上次获取信息，避免重复获取
  - 可勾选"强制重新获取"忽略缓存
  - 文本数据支持多线程并发爬取（默认3路并发，可在 config.py 中调整 `CRAWLER_MAX_WORKERS`）
- **双模型对比**：选择 Fin-Transformer 时，预测完成后自动加载 Transformer 基线模型进行对比推理，生成三线对比走势图（实际收盘价 / Fin-Transformer 预测 / Transformer 预测）
- **SHAP 可解释性**：预测后自动生成 SHAP 特征重要性图（summary plot + beeswarm + violin plot）
- **退出**：点击侧边栏"退出应用"可安全关闭

### 单股训练

```bash
python main.py --mode train
```

### 单股评估

```bash
python main.py --mode eval
```

### 命令行预测

```bash
python predict_next.py
```

交互式选择股票进行预测，支持单选、多选（逗号分隔）或全选。预测完成后自动清理中间文件。

### 消融实验

```bash
# 全部训练
python experiments/ablation_train.py --stock all --model all

# 指定股票和模型，支持断点续跑
python experiments/ablation_train.py --stock 601288 --model FinTransformer --resume
```

### 五模型对比实验

```bash
# 全部训练
python experiments/comparison_train.py --stock all --model all

# 指定模型
python experiments/comparison_train.py --stock all --model LSTM

# 可视化对比结果
python experiments/plot_comparison.py
```

### 鲁棒性实验

```bash
# 窗口长度敏感性
python experiments/robustness_test.py --experiment window

# 随机种子稳定性
python experiments/robustness_test.py --experiment seed

# 全部
python experiments/robustness_test.py --experiment all
```

### 统计显著性检验

```bash
python experiments/statistical_tests.py --pred_dir experiments/results/predictions
```

执行 Wilcoxon 符号秩检验（逐股票比较 FinTransformer 与 SimpleTransformer 绝对误差，Bonferroni 校正）、Cohen's d 效应量计算，以及整体 MAPE 改善率单样本 t 检验。

### 单元测试

```bash
pytest tests/ -v
```

共 23 个测试用例，覆盖 4 种模型变体的前向传播形状、损失函数正确性、金融先验偏置输出范围、注意力权重非负性与归一化、归一化/反归一化对称性、文本缺失零向量填充、数据划分无重叠等。

## 6. 数据说明

### 双数据库分离

| 用途     | MongoDB 库   | 股价数据目录          | 时间范围 |
|----------|-------------|----------------------|----------|
| 训练数据 | `posts`     | `data/data/prices/`   | 2025年   |
| 预测数据 | `posts_test`| `data/data/prices_test/`| 2026年   |

### 数据获取

- **股价数据**：通过 [Tushare](https://tushare.pro/) API 获取日线行情
- **文本数据**：通过 Selenium 爬虫抓取东方财富股吧评论，包含反检测措施（stealth.min.js）、随机休眠模拟人类行为、超量帖子精简爬取策略

### FinBERT 微调

不使用原始预训练 FinBERT，而是基于人工标注的股吧数据集进行定向微调：

1. 标注数据：`models/FinBERT/train/train.csv`（三分类标签：-1 消极 / 0 中性 / 1 积极）
2. 微调代码：`models/FinBERT/train/train.py`
3. 输出模型：`models/FinBERT/finetuned_finbert_guba/`

### 文本处理流程

```
帖子文本 → 微调FinBERT情感分析 → 768维CLS嵌入 → 按交易日置信度加权平均 → 每日文本向量
```

聚合窗口：前一交易日收盘后（15:00）至当前交易日开盘前（09:30），包含周末和非交易时段的帖子。

## 7. 关键配置

`config.py` 中的核心参数：

| 参数 | 值 | 说明 |
|------|----|------|
| `SEQ_LENGTH` | 50 | 输入序列长度（交易日） |
| `PRED_LENGTH` | 1 | 预测步长 |
| `D_MODEL` | 128 | Transformer 隐藏维度 |
| `NHEAD` | 4 | 多头注意力头数 |
| `NUM_LAYERS` | 2 | Transformer 编码器层数 |
| `DIM_FEEDFORWARD` | 256 | 前馈网络维度 |
| `DROPOUT` | 0.1 | Dropout 概率 |
| `BATCH_SIZE` | 64 | 批大小 |
| `EPOCHS` | 120 | 训练轮数 |
| `LR` | 0.0005 | 学习率 |
| `FEATURE_COLS` | `open, high, low, close, vol, amount, pct_chg` | 7维数值特征 |
| `PRIOR_BIAS_SCALE` | 0.1 | 金融先验偏置初始缩放因子 |
| `ATTN_REG_ENTROPY` | 0.01 | 注意力熵正则化权重 |
| `ATTN_REG_SPARSE` | 0.001 | 注意力稀疏正则化权重 |
| `SHAP_N_SAMPLES` | 200 | SHAP 解释的测试样本数 |
| `CRAWLER_MAX_WORKERS` | 3 | 文本爬取最大并发数（建议2-5） |
| `CHROME_BINARY_PATH` | `drivers/chrome/chrome.exe` | Chrome 浏览器本地路径 |
| `CHROMEDRIVER_PATH` | `drivers/chromedriver/chromedriver.exe` | ChromeDriver 本地路径 |

### Fin-Transformer 与 SimpleTransformer 参数对比

| 参数 | Fin-Transformer | SimpleTransformer |
|------|----------------|-------------------|
| d_model | 128 | 64 |
| nhead | 4 | 2 |
| num_layers | 2 | 1 |
| dim_feedforward | 256 | 128 |
| epochs | 120 | 120 |
| lr | 0.0005 | 0.0005 |

### 对比模型参数

| 参数 | LSTM | BiLSTM-Attention | Informer |
|------|------|------------------|----------|
| d_model / hidden | 128 | 128 (hidden=64×2) | 128 |
| num_layers | 2 | 2 | 2 |
| 文本流 | ✗ | ✓ | ✗ |
| 特殊参数 | — | 时间注意力 | factor=5 (ProbSparse) |

## 8. 评估指标

### 传统指标

- RMSE（均方根误差）
- MAE（平均绝对误差）
- MAPE（平均绝对百分比误差）
- R²（决定系数）

### 金融指标

- 年化收益率：基于真实价格序列的持有期收益年化
- 夏普比率：基于预测价格的日收益率波动调整（假设无风险利率为0）
- 最大回撤

### 可解释性

- **SHAP GradientExplainer**：基于梯度的特征归因分析，输出各时间步、各特征维度对预测结果的贡献度
- **summary plot**：数值特征全局重要性柱状图 + 文本流时间步重要性曲线
- **beeswarm + violin plot**：数值特征 SHAP 值分布可视化

## 9. 常见问题

**Q: 模型文件不存在？**
A: 先运行 `python main.py --mode train` 训练模型，或从备份中恢复模型文件到对应路径。

**Q: 文本数据获取失败？**
A: 检查 MongoDB 是否运行（`mongod`），Chrome 浏览器和 ChromeDriver 路径是否正确，网络是否能访问东方财富股吧。

**Q: SHAP 分析过慢？**
A: 调整 `config.py` 中的 `SHAP_N_SAMPLES`（默认200）和 `SHAP_BACKGROUND_SIZE`（默认50）。

**Q: 爬虫被封 IP？**
A: 降低 `config.py` 中的 `CRAWLER_MAX_WORKERS`（默认3），系统已有随机休眠和反检测措施。

**Q: 中文图表乱码？**
A: 确保系统安装了 SimHei 或 Microsoft YaHei 字体。`src/utils.py` 中的 `setup_chinese_font()` 会自动匹配系统可用字体。
