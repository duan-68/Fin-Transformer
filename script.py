# 脚本文件，用于处理各种零碎任务

# # 脚本：清除mongodb集合中的所有文档
# from pymongo import MongoClient, ASCENDING
# client = MongoClient('mongodb://localhost:27017/')
# db = client['posts']  

# # 假设你已经连接到数据库，collection 是你的集合对象
# collection = db['post_601288']
# collection.delete_many({})
# print("已清空 post_601288 集合中的所有文档。")


# 脚本：将指定文件夹中的所有 JSON 文件导入 MongoDB
import json
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

# ========== 配置 ==========
JSON_FILE = r"E:\code\project\data\data\posts\posts.post_601288.json"
MONGO_HOST = "localhost"
MONGO_PORT = 27017
DATABASE_NAME = "posts"
COLLECTION_NAME = "post_601288"
# ==========================

client = MongoClient(MONGO_HOST, MONGO_PORT)
collection = client[DATABASE_NAME][COLLECTION_NAME]

with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 统一转为列表
if isinstance(data, dict):
    data = [data]
elif not isinstance(data, list):
    raise ValueError("JSON 文件内容格式不支持")

try:
    if data:
        result = collection.insert_many(data, ordered=False)
        print(f"成功导入 {len(result.inserted_ids)} 条文档")
except BulkWriteError as e:
    # 忽略 _id 重复的错误，其他错误可继续查看
    inserted = e.details.get('nInserted', 0)
    dup_count = sum(1 for err in e.details.get('writeErrors', []) if err.get('code') == 11000)
    print(f"导入完成：成功 {inserted} 条，{dup_count} 条因重复键跳过")

client.close()