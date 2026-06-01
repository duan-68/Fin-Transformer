"""
配置文件
集中管理项目参数、路径和股票选择。
"""

import os

# ================= 项目根目录 =================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= 数据路径 =================
DATA_DIR = os.path.join(ROOT_DIR, 'data')
PRICE_DATA_DIR = os.path.join(DATA_DIR, 'data', 'prices')       # 原始股价数据目录
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')             # 处理后的文本向量存储目录
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
SENTIMENT_DIR = os.path.join(REPORTS_DIR, 'sentiment_analysis')
EMBEDDINGS_DIR = os.path.join(SENTIMENT_DIR, 'embeddings')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# 确保必要目录存在
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SENTIMENT_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ================= Selenium 浏览器配置 =================
CHROME_BINARY_PATH = os.path.join(ROOT_DIR, 'drivers', 'chrome', 'chrome.exe')
CHROMEDRIVER_PATH = os.path.join(ROOT_DIR, 'drivers', 'chromedriver', 'chromedriver.exe')

# ================= 爬虫并发配置 =================
CRAWLER_MAX_WORKERS = 3  # 文本数据爬取最大并发数（建议2-5，过高有封IP风险）

# ================= MongoDB 配置 =================
MONGO_HOST = 'localhost'
MONGO_PORT = 27017
DB_NAME = 'posts'

# ================= FinBERT 模型路径 =================
FINBERT_MODEL_PATH = os.path.join(MODELS_DIR, 'FinBERT', 'finetuned_finbert_guba')

# ================= 股票相关配置 =================
# 目标股票代码
STOCK_CODE = '601288'
# 股价数据中股票代码的后缀（如 '.SH'、'.SZ'），留空则自动匹配
STOCK_SUFFIX = '.SH'

# ================= 市场时间定义 =================
MARKET_CLOSE_TIME = "15:00"
MARKET_OPEN_TIME = "09:30"

# ================= 特征与目标 =================
FEATURE_COLS = ['open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg']
TARGET_COL = 'close'

# ================= 模型超参数 =================
SEQ_LENGTH = 50           # 回溯窗口长度
PRED_LENGTH = 1           # 预测未来天数
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

# =================可解释性分析配置=================
SHAP_N_SAMPLES = 200          # 用于 SHAP 解释的测试样本数
SHAP_BACKGROUND_SIZE = 50     # 背景样本数

# ================= 训练配置 =================
BATCH_SIZE = 64
EPOCHS = 120
LR = 0.0005
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

# ================= 金融先验配置 =================
PRIOR_BIAS_SCALE = 0.1      # 先验偏置初始缩放因子
ATTN_REG_ENTROPY = 0.01     # 注意力熵正则化权重
ATTN_REG_SPARSE = 0.001     # 注意力稀疏正则化权重

# ================= 消融实验配置 =================
ABLATION_MODELS = ['SimpleTransformer', 'ConcatFusion', 'CrossAttnOnly', 'FinTransformer']
ABLATION_MODEL_DIR = 'models/ablation'
ABLATION_RESULTS_DIR = 'experiments/results'

# ================= 随机种子 =================
SEED = 42

# ================= 动态生成路径函数 =================
def get_text_vec_path(stock_code):
    """获取每日文本向量文件路径"""
    stock_dir = os.path.join(PROCESSED_DIR, stock_code)
    return os.path.join(stock_dir, 'daily_text_vectors.npy')

def get_trading_days_path(stock_code):
    """获取交易日列表文件路径"""
    stock_dir = os.path.join(PROCESSED_DIR, stock_code)
    return os.path.join(stock_dir, 'trading_days.npy')

def get_sentiment_csv_path(stock_code):
    """获取情感分析结果 CSV 路径"""
    return os.path.join(SENTIMENT_DIR, f'{stock_code}_sentiment_results.csv')

def get_embeddings_dir(stock_code):
    """获取嵌入向量存储目录"""
    return os.path.join(EMBEDDINGS_DIR, stock_code)

def get_model_save_path(stock_code, model_type='fin-transformer'):
    """获取训练好的模型保存路径
    
    Args:
        stock_code: 股票代码
        model_type: 模型类型，'fin-transformer' 或 'transformer'
    """
    if model_type == 'fin-transformer':
        return os.path.join(MODELS_DIR, 'Fin-Transformer', f'Fin-Transformer_{stock_code}.pth')
    elif model_type == 'transformer':
        return os.path.join(MODELS_DIR, 'Transformer', f'Transformer_{stock_code}.pth')
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

def get_figure_save_path(stock_code):
    """获取预测结果图片保存路径"""
    return os.path.join(FIGURES_DIR, f'Fin-Transformer_result_{stock_code}.png')

# 股票列表
STOCK_LIST = [
    {'code': '601288', 'name': '农业银行', 'suffix': '.SH'},
    {'code': '601857', 'name': '中国石油', 'suffix': '.SH'},
    {'code': '601398', 'name': '工商银行', 'suffix': '.SH'},
    {'code': '600519', 'name': '贵州茅台', 'suffix': '.SH'},
    {'code': '300750', 'name': '宁德时代', 'suffix': '.SZ'},
    {'code': '601988', 'name': '中国银行', 'suffix': '.SH'},
    {'code': '601138', 'name': '工业富联', 'suffix': '.SH'},
    {'code': '601628', 'name': '中国人寿', 'suffix': '.SH'},
    {'code': '600036', 'name': '招商银行', 'suffix': '.SH'},
    {'code': '601899', 'name': '紫金矿业', 'suffix': '.SH'},
    {'code': '601088', 'name': '中国神华', 'suffix': '.SH'},
    {'code': '601318', 'name': '中国平安', 'suffix': '.SH'},
    {'code': '600900', 'name': '长江电力', 'suffix': '.SH'},
    {'code': '600028', 'name': '中国石化', 'suffix': '.SH'},
    {'code': '300308', 'name': '中际旭创', 'suffix': '.SZ'},
    {'code': '688041', 'name': '海光信息', 'suffix': '.SH'},
    {'code': '000333', 'name': '美的集团', 'suffix': '.SZ'},
    {'code': '688256', 'name': '寒武纪',   'suffix': '.SH'},
    {'code': '601728', 'name': '中国电信', 'suffix': '.SH'},
    {'code': '000858', 'name': '五粮液',   'suffix': '.SZ'},
]