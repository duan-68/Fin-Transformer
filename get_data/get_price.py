# 股票数据采集（Tushare API）
import tushare as ts
import pandas as pd
import time
import os
from datetime import datetime, timedelta


TOKEN = '3cf53053a14be407972da15204224f40125b58060686ae45c61091c5'
# 数据起止日期(动态计算近90天)
DAYS = 90
END_DATE   = datetime.now().strftime('%Y%m%d')
START_DATE = (datetime.now() - timedelta(days=DAYS)).strftime('%Y%m%d')
# 预设的存储股票列表的csv
INPUT_CSV = 'data/top20_stock.csv'  
# 输出目录(保存日线数据)
OUTPUT_DIR = 'data/data/prices_test'
# 是否合并为一个文件(True 则保存为单个CSV,False 则每只股票单独保存)
MERGE_OUTPUT = False


_pro = None

def _get_pro():
    """获取 tushare pro API 实例（单例模式）"""
    global _pro
    if _pro is None:
        ts.set_token(TOKEN)
        _pro = ts.pro_api()
    return _pro


def add_suffix(code):
    """根据代码前缀添加市场后缀"""
    if code.startswith(('6', '688')):
        return code + '.SH'
    elif code.startswith(('0', '3')):
        return code + '.SZ'
    else:
        return code + '.UNKNOWN'  # 未知市场,可手动处理


def fetch_stock_price(stock_code, suffix, start_date, end_date, output_dir):
    """获取单只股票价格数据并保存 CSV，返回是否成功。"""
    try:
        pro = _get_pro()
        ts_code = f"{stock_code}{suffix}"

        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

        if df is None or df.empty:
            print(f"警告:{ts_code} 没有数据")
            return False

        # 添加股票代码列
        df['stock_code'] = ts_code

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{stock_code}.csv")
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"已保存:{output_file}")
        return True

    except Exception as e:
        print(f"获取 {stock_code}{suffix} 失败:{e}")
        return False


if __name__ == "__main__":
    # 初始化Tushare
    pro = _get_pro()
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 读取股票列表
    df_stocks = pd.read_csv(INPUT_CSV, sep='\t', encoding='utf-8')
    # 提取code列,并转换为字符串(保留前导0)
    codes_list = df_stocks['code'].astype(str).str.zfill(6)  # 确保6位数字

    codes_with_suffix = [add_suffix(code) for code in codes_list]

    print(f"共 {len(codes_with_suffix)} 只股票,开始获取日线数据...")

    # 用于存放所有数据的列表(如果MERGE_OUTPUT为True)
    all_data = []

    # 循环获取每只股票的数据
    for i, ts_code in enumerate(codes_with_suffix, 1):
        print(f"正在处理 [{i}/{len(codes_with_suffix)}]: {ts_code}")
        try:
            # 调用Tushare日线接口
            df = pro.daily(ts_code=ts_code,
                           start_date=START_DATE,
                           end_date=END_DATE)

            if df.empty:
                print(f"警告:{ts_code} 没有数据,跳过")
                continue

            # 添加股票代码列(便于合并)
            df['stock_code'] = ts_code

            if MERGE_OUTPUT:
                all_data.append(df)
            else:
                # 单独保存为CSV,文件名用原始6位代码
                output_file = os.path.join(OUTPUT_DIR, f"{codes_list[i-1]}.csv")
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
                print(f"已保存:{output_file}")

            # 控制请求频率(免费版每分钟最多200次,这里每请求一次暂停0.5秒,远低于限制)
            time.sleep(0.5)

        except Exception as e:
            print(f"获取 {ts_code} 失败:{e}")

    # 如果需要合并输出
    if MERGE_OUTPUT and all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        merged_file = os.path.join(OUTPUT_DIR, 'all_stocks_daily.csv')
        merged_df.to_csv(merged_file, index=False, encoding='utf-8-sig')
        print(f"所有数据已合并保存至:{merged_file}")

    print("全部处理完成！")

