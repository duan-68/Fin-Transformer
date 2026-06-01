"""
数据处理模块：股价加载、FinBERT 推理、文本聚合、样本构建
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from pymongo import MongoClient
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from tqdm import tqdm

import config
from .utils import set_seed

set_seed(config.SEED)

# ================= 股价数据加载 =================
def load_price_data(data_dir=None, stock_code=None):
    """
    加载所有股票或指定股票的股价数据。
    若 data_dir 是目录，则读取目录下所有 CSV 文件；
    若 data_dir 是文件，则读取单个文件。
    若指定 stock_code，则仅返回该股票的数据（同时自动匹配后缀）。
    """
    if data_dir is None:
        data_dir = config.PRICE_DATA_DIR

    if os.path.isdir(data_dir):
        all_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not all_files:
            raise FileNotFoundError(f"未找到任何 CSV 文件于目录: {data_dir}")
        df_list = []
        for file in all_files:
            try:
                df = pd.read_csv(file)
                # 提取股票代码
                if 'ts_code' in df.columns:
                    code = str(df['ts_code'].iloc[0])
                else:
                    code = os.path.basename(file).replace('.csv', '')
                df['stock_id'] = code
                df_list.append(df)
            except Exception as e:
                print(f"警告: 读取文件 {file} 失败: {e}")
        full_df = pd.concat(df_list, ignore_index=True)
    else:
        # 单文件模式
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"文件不存在: {data_dir}")
        full_df = pd.read_csv(data_dir)
        if 'stock_id' not in full_df.columns:
            full_df['stock_id'] = full_df['ts_code'].astype(str)

    full_df['trade_date'] = pd.to_datetime(full_df['trade_date'], format='%Y%m%d', errors='coerce')
    if full_df['trade_date'].isna().any():
        print(f"警告: 有 {full_df['trade_date'].isna().sum()} 条记录的 trade_date 无法解析，将被删除。")
        full_df = full_df.dropna(subset=['trade_date'])
    full_df = full_df.sort_values(['stock_id', 'trade_date']).reset_index(drop=True)

    if stock_code is not None:
        mask = full_df['stock_id'].str.contains(stock_code, regex=False)
        full_df = full_df[mask].copy()
        if len(full_df) == 0:
            raise ValueError(f"未找到股票代码包含 '{stock_code}' 的数据")

    print(f"[Data] 加载股价数据: {len(full_df)} 行, {full_df['stock_id'].nunique()} 只股票")
    return full_df


# ================= FinBERT 推理与嵌入提取 =================
def run_finbert_inference(stock_code, mongo_host=None, mongo_port=None, db_name=None,
                          model_path=None, batch_size=64, max_len=512,
                          date_start=None, date_end=None, progress_callback=None):
    """
    对 MongoDB 中指定股票的帖子进行情感分析，并保存嵌入向量。
    返回结果 DataFrame。
    """
    if mongo_host is None:
        mongo_host = config.MONGO_HOST
    if mongo_port is None:
        mongo_port = config.MONGO_PORT
    if db_name is None:
        db_name = config.DB_NAME
    if model_path is None:
        model_path = config.FINBERT_MODEL_PATH

    collection_name = f'post_{stock_code}'
    output_csv = config.get_sentiment_csv_path(stock_code)
    output_emb_dir = config.get_embeddings_dir(stock_code)
    os.makedirs(output_emb_dir, exist_ok=True)

    # 连接 MongoDB
    client = MongoClient(mongo_host, mongo_port)
    db = client[db_name]
    collection = db[collection_name]
    query_filter = {}
    if date_start or date_end:
        date_cond = {}
        if date_start:
            date_cond['$gte'] = date_start
        if date_end:
            date_cond['$lte'] = date_end
        query_filter['post_date'] = date_cond
        print(f"日期过滤: {date_cond}")
    data = list(collection.find(query_filter))
    if not data:
        print(f"集合 {collection_name} 无数据，跳过 FinBERT 推理。")
        return None
    print(f"股票 {stock_code}: 共读取到 {len(data)} 条帖子")

    # 加载模型
    device = torch.device(config.DEVICE)
    print(f"使用设备: {device}")
    config_bert = BertConfig.from_pretrained(model_path, num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, config=config_bert)
    model.to(device)
    model.eval()

    sentiment_labels = ["负面", "中性", "正面"]

    # 自定义 Dataset
    class PostDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    def collate_dicts(batch):
        return batch

    dataset = PostDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_dicts)

    results = []
    total_valid = 0
    total_items = len(data)
    current_processed = 0
    for batch in tqdm(dataloader, desc="FinBERT 推理"):
        current_processed += len(batch)
        if progress_callback:
            progress_callback(min(current_processed, total_items), total_items)
        valid_titles = []
        valid_docs = []
        for doc in batch:
            if isinstance(doc, dict) and doc.get('post_title'):
                valid_titles.append(doc['post_title'])
                valid_docs.append(doc)

        if not valid_titles:
            continue

        total_valid += len(valid_titles)
        inputs = tokenizer(valid_titles, return_tensors="pt", truncation=True,
                           padding=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            confidences = probs.max(dim=-1)[0].cpu().numpy()
            embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()

        for i, doc in enumerate(valid_docs):
            _id = doc.get('_id')
            sentiment = sentiment_labels[pred_ids[i]]
            confidence = round(confidences[i], 4)
            emb = embeddings[i]

            emb_path = os.path.join(output_emb_dir, f"{_id}.npy")
            np.save(emb_path, emb)

            result = {
                '_id': _id,
                'post_title': doc.get('post_title', ''),
                'post_date': doc.get('post_date'),
                'post_time': doc.get('post_time'),
                'sentiment': sentiment,
                'confidence': confidence,
                'emb_path': emb_path
            }
            results.append(result)

    print(f"有效帖子数: {total_valid}")
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding='utf_8_sig')
        print(f"情感结果已保存至 {output_csv}")
        print(f"嵌入向量已保存至 {output_emb_dir}")
        return df
    else:
        print("没有有效帖子，无法生成结果。")
        return None


# ================= 文本聚合（每日加权平均） =================
def aggregate_daily_text_vectors(stock_code, price_df=None):
    """
    读取情感分析结果 CSV，按交易日进行置信度加权平均，生成每日文本向量。
    保存 daily_text_vectors.npy 和 trading_days.npy。
    返回 daily_vecs 和 valid_days。
    """
    sentiment_csv = config.get_sentiment_csv_path(stock_code)
    if not os.path.exists(sentiment_csv):
        print(f"情感结果文件 {sentiment_csv} 不存在，请先运行 FinBERT 推理。")
        return None, None

        # 获取交易日列表
    if price_df is None:
        price_df = load_price_data(stock_code=stock_code)

    if price_df['stock_id'].nunique() == 1:
        stock_price = price_df.copy()
    else:
        stock_price = price_df[price_df['stock_id'].str.contains(stock_code, regex=False, na=False)]
    
    if len(stock_price) == 0:
        raise ValueError(f"股价数据中未找到股票 {stock_code}")
    
    print(f"股价数据样例:\n{stock_price[['trade_date']].head()}")
    trading_days = sorted(stock_price['trade_date'].dt.strftime('%Y%m%d').unique())
    print(f"股票 {stock_code} 交易日数量: {len(trading_days)}")

    # 读取情感结果
    df_sent = pd.read_csv(sentiment_csv)
    df_sent['datetime'] = pd.to_datetime(df_sent['post_date'].astype(str) + ' ' + df_sent['post_time'].astype(str),
                                         format='%Y%m%d %H:%M', errors='coerce')
    df_sent = df_sent.dropna(subset=['datetime'])
    print(f"有效帖子数（含时间）: {len(df_sent)}")

    # 加载嵌入向量的辅助函数
    def load_embedding(emb_path):
        try:
            return np.load(emb_path)
        except Exception:
            return np.zeros(768)

    daily_vectors = []
    valid_days = []
    for i in range(len(trading_days) - 1):
        day_t = trading_days[i]
        day_next = trading_days[i+1]
        start_dt = datetime.strptime(f"{day_t} {config.MARKET_CLOSE_TIME}", "%Y%m%d %H:%M")
        end_dt = datetime.strptime(f"{day_next} {config.MARKET_OPEN_TIME}", "%Y%m%d %H:%M")

        mask = (df_sent['datetime'] >= start_dt) & (df_sent['datetime'] < end_dt)
        df_window = df_sent[mask]

        if len(df_window) == 0:
            daily_vec = np.zeros(768)
        else:
            sum_weighted = np.zeros(768)
            sum_conf = 0.0
            for _, row in df_window.iterrows():
                emb = load_embedding(row['emb_path'])
                conf = row['confidence']
                sum_weighted += conf * emb
                sum_conf += conf
            daily_vec = sum_weighted / sum_conf if sum_conf > 0 else np.zeros(768)

        daily_vectors.append(daily_vec)
        valid_days.append(day_t)

    daily_vectors = np.array(daily_vectors)
    valid_days = np.array(valid_days)

    # 保存
    stock_dir = os.path.join(config.PROCESSED_DIR, stock_code)
    os.makedirs(stock_dir, exist_ok=True)
    np.save(config.get_text_vec_path(stock_code), daily_vectors)
    np.save(config.get_trading_days_path(stock_code), valid_days)
    print(f"每日文本向量已保存，共 {len(daily_vectors)} 天。")
    return daily_vectors, valid_days


# ================= 加载文本向量 =================
def load_text_vectors(stock_code):
    vec_file = config.get_text_vec_path(stock_code)
    days_file = config.get_trading_days_path(stock_code)
    if not os.path.exists(vec_file) or not os.path.exists(days_file):
        print(f"文本向量文件不存在，尝试生成...")
        daily_vecs, valid_days = aggregate_daily_text_vectors(stock_code)
        if daily_vecs is None:
            return None, None
        return daily_vecs, valid_days
    daily_vecs = np.load(vec_file)
    valid_days = np.load(days_file, allow_pickle=True)
    # 检查是否为空
    if len(daily_vecs) == 0:
        print(f"错误: 已存在的文本向量文件为空 (0天)，请删除后重新生成或检查数据。")
        return None, None
    print(f"[Data] 加载文本向量: {daily_vecs.shape[0]} 天, 维度 {daily_vecs.shape[1]}")
    return daily_vecs, valid_days


# ================= 样本对齐 =================
class DualStreamDataset(Dataset):
    def __init__(self, num_seqs, text_seqs, targets, stock_ids):
        self.num_seqs = torch.FloatTensor(num_seqs)
        self.text_seqs = torch.FloatTensor(text_seqs)
        self.targets = torch.FloatTensor(targets)
        self.stock_ids = stock_ids

    def __len__(self):
        return len(self.num_seqs)

    def __getitem__(self, idx):
        return self.num_seqs[idx], self.text_seqs[idx], self.targets[idx], self.stock_ids[idx]


def create_dual_sequences(df, daily_text_vecs, text_valid_days, feature_cols, target_col,
                          seq_length, pred_length, train_ratio, val_ratio):
    """
    为每只股票构建数值序列和对应的文本序列。
    若某日无对应文本向量，则用零向量填充。
    """
    text_vec_dim = daily_text_vecs.shape[1]
    date_to_vec = {day: vec for day, vec in zip(text_valid_days, daily_text_vecs)}

    train_num, train_text, train_tgt, train_ids = [], [], [], []
    val_num, val_text, val_tgt, val_ids = [], [], [], []
    test_num, test_text, test_tgt, test_ids = [], [], [], []

    scalers_dict = {}

    grouped = df.groupby('stock_id')
    for stock_id, group_df in grouped:
        group_df = group_df.sort_values('trade_date').reset_index(drop=True)
        n = len(group_df)
        if n < seq_length + pred_length + 10:
            continue

        data_raw = group_df[feature_cols].values.astype(np.float32)
        close_raw = group_df[target_col].values.astype(np.float32)

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data_raw)
        scalers_dict[stock_id] = {
            'close_min': close_raw.min(),
            'close_range': close_raw.max() - close_raw.min()
        }

        close_idx = feature_cols.index(target_col)
        dates = group_df['trade_date'].dt.strftime('%Y%m%d').values

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        limit = n - seq_length - pred_length + 1
        for i in range(limit):
            num_seq = data_scaled[i:i+seq_length]
            window_dates = dates[i:i+seq_length]
            text_seq = []
            for d in window_dates:
                if d in date_to_vec:
                    text_seq.append(date_to_vec[d])
                else:
                    text_seq.append(np.zeros(text_vec_dim))
            text_seq = np.array(text_seq)

            target_val = data_scaled[i + seq_length, close_idx]

            if i + seq_length < train_end:
                bucket = (train_num, train_text, train_tgt, train_ids)
            elif i + seq_length < val_end:
                bucket = (val_num, val_text, val_tgt, val_ids)
            else:
                bucket = (test_num, test_text, test_tgt, test_ids)

            bucket[0].append(num_seq)
            bucket[1].append(text_seq)
            bucket[2].append(target_val)
            bucket[3].append(stock_id)

    print(f"[Data] 样本构建: Train {len(train_num)}, Val {len(val_num)}, Test {len(test_num)}")
    return (np.array(train_num), np.array(train_text), np.array(train_tgt), np.array(train_ids),
            np.array(val_num), np.array(val_text), np.array(val_tgt), np.array(val_ids),
            np.array(test_num), np.array(test_text), np.array(test_tgt), np.array(test_ids),
            scalers_dict)


# ================= 纯数值序列数据集 =================
class NumOnlyDataset(Dataset):
    """纯数值序列数据集"""

    def __init__(self, num_seqs, targets, stock_ids):
        self.num_seqs = torch.FloatTensor(np.array(num_seqs))
        self.targets = torch.FloatTensor(np.array(targets))
        self.stock_ids = stock_ids

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.num_seqs[idx], self.targets[idx], self.stock_ids[idx]


def create_num_only_sequences(df, feature_cols, target_col, seq_length, pred_length, train_ratio, val_ratio):
    """构建纯数值序列数据集（无文本流），按时间顺序划分训练/验证/测试集。"""
    train_num, train_tgt, train_ids = [], [], []
    val_num, val_tgt, val_ids = [], [], []
    test_num, test_tgt, test_ids = [], [], []

    scalers_dict = {}

    grouped = df.groupby('stock_id')
    for stock_id, group_df in grouped:
        group_df = group_df.sort_values('trade_date').reset_index(drop=True)
        n = len(group_df)
        if n < seq_length + pred_length + 10:
            print(f"[Data] 股票 {stock_id} 数据量不足 ({n} 行)，跳过")
            continue

        data_raw = group_df[feature_cols].values.astype(np.float32)
        close_raw = group_df[target_col].values.astype(np.float32)

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data_raw)
        scalers_dict[stock_id] = {
            'close_min': float(close_raw.min()),
            'close_range': float(close_raw.max() - close_raw.min())
        }

        close_idx = feature_cols.index(target_col)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        limit = n - seq_length - pred_length + 1
        for i in range(limit):
            num_seq = data_scaled[i:i + seq_length]
            target_val = data_scaled[i + seq_length, close_idx]

            if i + seq_length < train_end:
                bucket = (train_num, train_tgt, train_ids)
            elif i + seq_length < val_end:
                bucket = (val_num, val_tgt, val_ids)
            else:
                bucket = (test_num, test_tgt, test_ids)

            bucket[0].append(num_seq)
            bucket[1].append(target_val)
            bucket[2].append(stock_id)

    print(f"[Data] 纯数值样本构建: Train {len(train_num)}, Val {len(val_num)}, Test {len(test_num)}")
    return (np.array(train_num) if train_num else np.array([]).reshape(0, seq_length, len(feature_cols)),
            np.array(train_tgt), np.array(train_ids),
            np.array(val_num) if val_num else np.array([]).reshape(0, seq_length, len(feature_cols)),
            np.array(val_tgt), np.array(val_ids),
            np.array(test_num) if test_num else np.array([]).reshape(0, seq_length, len(feature_cols)),
            np.array(test_tgt), np.array(test_ids),
            scalers_dict)