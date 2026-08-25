import pandas as pd
from pymongo import MongoClient, ASCENDING
from datetime import datetime, timedelta
import bisect

def delete_outside_range(collection, start="202501010000", end="202512312359"):
    """删除集合中不在指定时间范围内的数据"""

    # 构造聚合表达式，生成可比较的字符串 "post_date" + "post_time"（去掉冒号）
    combined_datetime = {
        "$concat": [
            "$post_date",
            {
                "$replaceAll": {
                    "input": "$post_time",
                    "find": ":",
                    "replacement": ""
                }
            }
        ]
    }

    # 筛选条件：不在 [start, end] 区间内
    filter_outside = {
        "$expr": {
            "$not": {
                "$and": [
                    { "$gte": [ combined_datetime, start ] },
                    { "$lte": [ combined_datetime, end ] }
                ]
            }
        }
    }

    # 预览要删除的文档数量
    to_delete = collection.count_documents(filter_outside)
    print(f"文档总数：{collection.count_documents({})}")
    print(f"将要删除的文档数：{to_delete}")

    if to_delete > 0:
        # 执行删除
        result = collection.delete_many(filter_outside)
        print(f"成功删除 {result.deleted_count} 条文档\n")
    else:
        print("没有需要删除的文档\n")


def get_F_L_by_time(collection, stock=""):
    """返回集合中按 post_date 和 post_time 组合时间排序的第一个和最后一个文档。"""
    # 定义组合字段的表达式（用于排序）
    pipeline_sort = {
        "$addFields": {
            "full_datetime": {
                "$concat": [
                    "$post_date",
                    {
                        "$replaceAll": {
                            "input": "$post_time",
                            "find": ":",
                            "replacement": ""
                        }
                    }
                ]
            }
        }
    }

    # 获取第一条（最早时间）：按 full_datetime 升序，取1
    first_cursor = collection.aggregate([
        pipeline_sort,
        {"$sort": {"full_datetime": 1}},
        {"$limit": 1}
    ])
    first_list = list(first_cursor)
    first_doc = first_list[0] if first_list else None

    # 获取最后一条（最晚时间）：按 full_datetime 降序，取1
    last_cursor = collection.aggregate([
        pipeline_sort,
        {"$sort": {"full_datetime": -1}},
        {"$limit": 1}
    ])
    last_list = list(last_cursor)
    last_doc = last_list[0] if last_list else None


    print(f"集合post_{stock}中第一条数据为：{first_doc}\n")
    print(f"集合post_{stock}中最后一条数据为：{last_doc}")
    print("--------------------------------------------------------")
    return first_doc, last_doc

def is_sorted_by_time(collection, allow_equal=True):
    """
    检查集合中的文档是否按 post_date+post_time 组合时间严格递增。
    
    参数:
        collection: pymongo 集合对象
        allow_equal: 是否允许相邻文档时间相等（True表示允许非递减，False表示必须严格递增）
    
    返回:
        (is_sorted, total_count, disorder_count, examples)
        - is_sorted: 布尔值，表示是否符合顺序要求
        - total_count: 总文档数
        - disorder_count: 不满足顺序的相邻文档对数
        - examples: 前3个乱序示例（每个示例为(前一个文档_id, 后一个文档_id, 前时间, 后时间)）
    """
    # 定义聚合表达式，生成组合时间字符串 "YYYYMMDDHHMM"
    pipeline = [
        {
            "$addFields": {
                "full_datetime": {
                    "$concat": [
                        "$post_date",
                        {
                            "$replaceAll": {
                                "input": "$post_time",
                                "find": ":",
                                "replacement": ""
                            }
                        }
                    ]
                }
            }
        },
        {"$sort": {"full_datetime": ASCENDING}},  # 按时间升序
        {"$project": {"_id": 1, "full_datetime": 1}}  # 只取必要字段
    ]

    cursor = collection.aggregate(pipeline, allowDiskUse=True)  # 允许磁盘使用，避免内存限制

    prev_doc = None
    total = 0
    disorder_count = 0
    examples = []

    for doc in cursor:
        total += 1
        if prev_doc is None:
            prev_doc = doc
            continue

        prev_time = prev_doc["full_datetime"]
        curr_time = doc["full_datetime"]

        # 比较
        if allow_equal:
            condition = prev_time > curr_time   # 允许相等，只有前 > 后才算乱序
        else:
            condition = prev_time >= curr_time  # 不允许相等，前 >= 后即乱序

        if condition:
            disorder_count += 1
            if len(examples) < 3:
                examples.append({
                    "prev_id": prev_doc["_id"],
                    "curr_id": doc["_id"],
                    "prev_time": prev_time,
                    "curr_time": curr_time
                })
        prev_doc = doc

    is_sorted = (disorder_count == 0)
    print(is_sorted, total, disorder_count, examples)
    return is_sorted, total, disorder_count, examples

def fill_missing_dates(collection, stock, start_date="20250101", end_date="20251231"):
    """
    检查集合在指定日期范围内每一天是否有数据，若缺失则插入默认文档，
    并根据前后文档的_id线性插值计算新_id，保证唯一性。
    """
    # 生成范围内的所有日期
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    delta = end - start
    all_dates = [(start + timedelta(days=i)).strftime("%Y%m%d") for i in range(delta.days + 1)]

    # 查询范围内所有文档，按日期和_id排序
    cursor = collection.find(
        {"post_date": {"$gte": start_date, "$lte": end_date}},
        {"_id": 1, "post_date": 1}
    ).sort([("post_date", 1), ("_id", 1)])
    docs = list(cursor)

    if not docs:
        print(f"集合 {stock} 在范围内没有数据，无法补全（缺少参考点）")
        print("-" * 50)
        return

    # 按日期分组，取每个日期最小的_id作为代表点（用于插值）
    date_to_min_id = {}
    for doc in docs:
        date = doc["post_date"]
        if date not in date_to_min_id or doc["_id"] < date_to_min_id[date]:
            date_to_min_id[date] = doc["_id"]
    sorted_dates = sorted(date_to_min_id.keys())
    sorted_ids = [date_to_min_id[d] for d in sorted_dates]

    # 计算缺失日期
    existing_dates_set = set(date_to_min_id.keys())
    missing_dates = sorted(set(all_dates) - existing_dates_set)
    if not missing_dates:
        print(f"集合 {stock} 在 {start_date}~{end_date} 范围内每天都有数据，无需补全。")
        print("-" * 50)
        return

    print(f"集合 {stock} 缺失 {len(missing_dates)} 天数据，正在补全...")

    # 获取所有现有_id（用于唯一性检查）
    existing_ids = set(doc["_id"] for doc in docs)
    used_ids = set()  # 记录本次即将插入的_id

    new_docs = []
    for d in missing_dates:
        d_date = datetime.strptime(d, "%Y%m%d")
        # 在sorted_dates中二分查找位置
        pos = bisect.bisect_left(sorted_dates, d)
        prev_idx = pos - 1
        next_idx = pos

        # 根据前后情况计算理想_id
        if prev_idx >= 0 and next_idx < len(sorted_dates):
            # 前后都有
            d1 = sorted_dates[prev_idx]
            d2 = sorted_dates[next_idx]
            id1 = sorted_ids[prev_idx]
            id2 = sorted_ids[next_idx]
            date1 = datetime.strptime(d1, "%Y%m%d")
            date2 = datetime.strptime(d2, "%Y%m%d")
            total_days = (date2 - date1).days
            k = (d_date - date1).days
            new_id = round(id1 + (id2 - id1) * k / total_days)
        elif prev_idx >= 0 and next_idx == len(sorted_dates):
            # 只有前一个（缺失在结尾）
            if prev_idx > 0:
                # 用前一个和前前一个计算步长
                d_prev = sorted_dates[prev_idx-1]
                id_prev = sorted_ids[prev_idx-1]
                d_prev_date = datetime.strptime(d_prev, "%Y%m%d")
                d_curr_date = datetime.strptime(sorted_dates[prev_idx], "%Y%m%d")
                step = (sorted_ids[prev_idx] - id_prev) / (d_curr_date - d_prev_date).days
            else:
                step = 1  # 仅有一个点，默认步长1
            last_date = datetime.strptime(sorted_dates[prev_idx], "%Y%m%d")
            days_after = (d_date - last_date).days
            new_id = round(sorted_ids[prev_idx] + step * days_after)
        elif prev_idx < 0 and next_idx < len(sorted_dates):
            # 只有后一个（缺失在开头）
            if next_idx < len(sorted_dates)-1:
                # 用后一个和后一个之后的点计算步长
                d_next = sorted_dates[next_idx+1]
                id_next = sorted_ids[next_idx+1]
                d_next_date = datetime.strptime(sorted_dates[next_idx], "%Y%m%d")
                d_next2_date = datetime.strptime(d_next, "%Y%m%d")
                step = (id_next - sorted_ids[next_idx]) / (d_next2_date - d_next_date).days
            else:
                step = 1
            first_date = datetime.strptime(sorted_dates[next_idx], "%Y%m%d")
            days_before = (first_date - d_date).days
            new_id = round(sorted_ids[next_idx] - step * days_before)
        else:
            continue  # 理论上不会发生

        # 确保_id为正整数
        if new_id <= 0:
            new_id = 1

        # 处理_id冲突：若与现有或已生成冲突，则递增直到唯一
        while new_id in existing_ids or new_id in used_ids:
            new_id += 1
        used_ids.add(new_id)

        # 构建默认文档
        doc = {
            "_id": new_id,
            "post_title": "无标题",
            "post_view": "0",
            "comment_num": 0,
            "post_url": "无",
            "post_date": d,
            "post_time": "12:00",
            "post_author": "无"
        }
        new_docs.append(doc)

    # 批量插入
    if new_docs:
        try:
            result = collection.insert_many(new_docs, ordered=False)
            print(f"集合 {stock} 成功插入 {len(result.inserted_ids)} 条补全数据。")
            print("-" * 50)
        except Exception as e:
            print(f"集合 {stock} 插入失败：{e}")
            print("-" * 50)
    else:
        print(f"集合 {stock} 没有需要补全的数据。")
        print("-" * 50)

def count(codes_list):
    """获取每只股票数据量和所有股票数据总量"""
    total_docs = 0
    print("各集合文档数量：")
    for stock in codes_list:
        collection = db[f"post_{stock}"]
        count = collection.count_documents({})  # 精确计数，如需更快可用 estimated_document_count()
        print(f"股票{stock}: {count}条")
        total_docs += count
    print(f"\n数据库中所有股票集合文档总数:{total_docs}条")


if __name__ == "__main__":
    # 获取需要处理的股票代码列表
    df_stocks = pd.read_csv("data/top20_stock.csv",sep='\t', encoding='utf-8')
    codes_list = df_stocks['code'].astype(str).str.zfill(6)  # 确保6位数字

    # 连接MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['posts_test']          # 数据库
    
    # 数据处理
    for stock in codes_list:  
        collection = db[f"post_{stock}"]    # 集合
        print(stock,end="")    
        # delete_outside_range(collection, start="202602040000", end="202604242359")  # 删除不在指定时间范围内的数据
        # get_F_L_by_time(collection)    # 获取集合中第一条和最后一条数据
        # is_sorted_by_time(collection, allow_equal=True)  # 查看数据是否按照时间顺序排列
        # fill_missing_dates(collection, stock, start_date="20260204", end_date="20260424")  # 对缺失的数据进行处理
    
    # count(codes_list)
    client.close()





