import os
import re
import sys
import time
import random
import hashlib
import threading
import pandas as pd

from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

from pymongo import MongoClient
from pymongo.errors import BulkWriteError

# 将项目根目录加入 sys.path，以便导入 config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

DB = "posts_test"

class PostSpider(object):
    """爬取东方财富网股吧帖子爬虫类"""

    def __init__(self, stock_code: str):
        self.browser = None
        self.code = stock_code
        self.start = time.time()  # 计算爬取所用时间，此为开始时间

    def create_webdriver(self):
        """创建并配置webdriver,添加防反爬手段"""
        options = webdriver.ChromeOptions()  # 配置webdriver
        options.add_argument('lang=zh_CN.UTF-8')
        options.add_argument('user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                             'like Gecko) Chrome/111.0.0.0 Safari/537.36"')

        # 指定浏览器和驱动路径，避免自动发现失败
        options.binary_location = config.CHROME_BINARY_PATH
        service = Service(executable_path=config.CHROMEDRIVER_PATH)

        self.browser = webdriver.Chrome(service=service, options=options)

        current_dir = os.path.dirname(os.path.abspath(__file__))  # 隐藏爬虫的自动化(selenium)的特征
        js_file_path = os.path.join(current_dir, 'stealth.min.js')
        with open(js_file_path) as f:
            js = f.read()
        self.browser.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": js
        })

    def get_page_num(self):
        """获取股吧帖子总页数"""
        self.browser.get(f'http://guba.eastmoney.com/list,{self.code},f_1.html')
        page_element = self.browser.find_element(By.CSS_SELECTOR, 'ul.paging > li:nth-child(7) > a > span')
        return int(page_element.text)

    def spider_post_info(self, page1: int, page2: int, cutoff_date: str = None, target_start_date: str = None):
        """爬取从page1到page2页的帖子信息"""
        self.create_webdriver()
        max_page = self.get_page_num()  # 确认最大页码
        current_page = page1  # 起始页
        stop_page = min(page2, max_page)  # 避免超出索引
        
        page_num = stop_page - page1 + 1
        if page_num >=500:
            print(f"股票{self.code}需要爬取的帖子总页数过多，进行精简爬取！")   # 处理对应时间段帖子页数过多的股票
        
        parser = PostParser()  # 必须在循环外创建,因为它包含处理日期的状态变量
        postdb = MongoAPI(DB, f'post_{self.code}')  # 连接集合

        while current_page <= stop_page:
            time.sleep(abs(random.normalvariate(0, 0.1)))  # 随机休眠,模拟人类行为
            url = f'http://guba.eastmoney.com/list,{self.code},f_{current_page}.html'

            try:
                self.browser.get(url) 
                dic_list = []
                list_item = self.browser.find_elements(By.CSS_SELECTOR, '.listitem')  # 获取当前页所有帖子
                if current_page == 1:
                    list_item = list_item[1:]  # 剔除首页的置顶帖

                # 过量帖子的精简爬取，爬取之后睡眠一定时间，防止触发反爬
                if page_num >= 500 and page_num <600 :
                    list_item = list_item[:60]    # 500页，前60条，
                    time.sleep(0.75)
                elif page_num >= 600 and page_num <800 :
                    list_item = list_item[:50]    # 600页，前50条
                    time.sleep(1)
                elif page_num >= 800 and page_num <1000 :
                    list_item = list_item[::2]    # 800页，每两条取一条
                    time.sleep(1.6)
                elif page_num >= 1000:
                    list_item = list_item[::3]    # 大于等于1000页，每三条取一条
                    time.sleep(2)
                for li in list_item:  # 分别处理每个帖子
                    try:
                        dic = parser.parse_post_info(li)
                        if 'guba.eastmoney.com/news' in dic['post_url']:  # 其他网站格式不同,只保留股吧帖子
                            dic_list.append(dic)
                    except Exception as e:
                        print(f'{self.code}: 帖子解析失败: {e}，跳过')
                        continue
                # 插入前批量查重（用于重复率早停判断）
                duplicate_count = 0
                if target_start_date and dic_list:
                    id_list = [d['_id'] for d in dic_list]
                    existing_ids = set(
                        doc['_id'] for doc in postdb.collection.find(
                            {'_id': {'$in': id_list}}, {'_id': 1}
                        )
                    )
                    duplicate_count = len(existing_ids)

                postdb.insert_many(dic_list)
                print(f'{self.code}: 已经成功爬取第 {current_page} 页帖子基本信息,'
                      f'进度 {(current_page - page1 + 1)*100/(stop_page - page1 + 1):.2f}%')

                # 基于日期的即时停止检查
                if cutoff_date:
                    old_posts = sum(1 for d in dic_list if d.get('post_date') and d['post_date'] < cutoff_date)
                    if len(dic_list) > 0 and old_posts > len(dic_list) / 2:
                        print(f'{self.code}: 第 {current_page} 页超过半数帖子早于 {cutoff_date}，停止爬取')
                        break

                # 基于重复数据+覆盖范围的早停检查
                if target_start_date and dic_list:
                    duplicate_ratio = duplicate_count / len(dic_list)
                    if duplicate_ratio > 0.5:
                        earliest_doc = postdb.collection.find_one(
                            sort=[('post_date', 1)]
                        )
                        if earliest_doc and earliest_doc.get('post_date', '') <= target_start_date:
                            print(f'{self.code}: 第 {current_page} 页超过半数重复（{duplicate_count}/{len(dic_list)}），'
                                  f'且数据已覆盖至 {earliest_doc["post_date"]}（目标 {target_start_date}），停止爬取')
                            break
                        else:
                            print(f'{self.code}: 第 {current_page} 页有重复数据（{duplicate_count}/{len(dic_list)}），'
                                  f'但尚未覆盖完整范围，继续爬取')

                current_page += 1

            except Exception as e:
                print(f'{self.code}: 第 {current_page} 页出现了错误 {e}')
                time.sleep(0.01)
                self.browser.refresh()
                self.browser.delete_all_cookies()
                self.browser.quit()  
                self.create_webdriver()  # 重新启动！(如果不重启webdriver,访问速度会被限制)

        end = time.time()     # 计算耗时,此为结束时间
        time_cost = end - self.start  # 任务总耗时
        start_date = postdb.find_last()['post_date']
        end_date = postdb.find_first()['post_date']  # 获取帖子时间范围
        row_count = postdb.count_documents()  # 统计总条数
        self.browser.quit()

        print(f'成功爬取 {self.code}股吧共 {stop_page - page1 + 1} 页帖子,总计 {row_count} 条,花费 {time_cost/60:.2f} 分钟')
        print(f'帖子的时间范围从 {start_date} 到 {end_date}')


class MongoAPI(object):
    """MongoDB操作接口类"""

    def __init__(self, db_name: str, collection_name: str, host='localhost', port=27017):
        self.host = host
        self.port = port
        self.db_name = db_name
        self.collection = collection_name
        self.client = MongoClient(host=self.host, port=self.port)
        self.database = self.client[self.db_name]
        self.collection = self.database[self.collection]

    def insert_one(self, kv_dict):
        """插入单条文档"""
        self.collection.insert_one(kv_dict)

    def insert_many(self, li_dict):
        """批量插入文档,忽略重复键错误"""
        try:
            self.collection.insert_many(li_dict, ordered=False)
        except BulkWriteError:
            pass

    def find_one(self, query1, query2):
        """查询单条文档"""
        return self.collection.find_one(query1, query2)

    def find(self, query1, query2):
        """查询多条文档,返回游标"""
        return self.collection.find(query1, query2)

    def find_first(self):
        """按_id正序获取第一条文档(最早的)"""
        return self.collection.find_one(sort=[('_id', 1)])
    
    def find_last(self):
        """按_id倒序获取第一条文档(最新的)"""
        return self.collection.find_one(sort=[('_id', -1)])

    def count_documents(self):
        """统计集合中的文档总数"""
        return self.collection.count_documents({})

    def update_one(self, kv_dict):
        """更新或插入单条文档(upsert)"""
        self.collection.update_one(kv_dict, {'$set': kv_dict}, upsert=True)

    def drop(self):
        """删除整个集合"""
        self.collection.drop()


class PostParser(object):
    """帖子信息解析类"""

    def __init__(self):
        self.year = None   # 用于推算年份的当前年份
        self.month = 13     # 用于判断年份是否跨年的辅助变量
        self.id = 0

    @staticmethod
    def parse_post_title(html):
        """解析帖子标题"""
        title_element = html.find_element(By.CSS_SELECTOR, 'td:nth-child(3) > div')
        return title_element.text

    @staticmethod
    def parse_post_view(html):
        """解析帖子浏览量"""
        view_element = html.find_element(By.CSS_SELECTOR, 'td > div')
        return view_element.text  # 保留为字符串,因为可能包含"万"等字符

    @staticmethod
    def parse_comment_num(html):
        """解析评论数,处理特殊情况(如"万")"""
        num_element = html.find_element(By.CSS_SELECTOR, 'td:nth-child(2) > div')
        text = num_element.text.strip()

        if not text or text in ['—', '-']:
            return 0

        try:
            comment_num = int(text)
        except ValueError:
            try:
                # 处理带"万"的情况，如 "3.5万" → 35000
                num_str = text.replace('万', '').strip()
                if num_str:
                    comment_num = int(float(num_str) * 10000)
                else:
                    comment_num = 0
            except (ValueError, AttributeError):
                comment_num = 0
        return comment_num

    @staticmethod
    def parse_post_url(html):
        """解析帖子链接"""
        url_element = html.find_element(By.CSS_SELECTOR, 'td:nth-child(3) > div > a')
        return url_element.get_attribute('href')

    @staticmethod
    def remove_char(date_str):
        """使用正则表达式去掉所有汉字字符(处理日期中包含"修改","来自"字符的情况)"""
        cleaned_str = re.sub(r'[^\d\s:-]', '', date_str)
        return cleaned_str.strip()

    def get_post_year(self, html):
        """通过访问帖子详情页获取发帖时间"""
        post_url = self.parse_post_url(html)

        # 指定浏览器和驱动路径，避免自动发现失败
        options = webdriver.ChromeOptions()
        options.binary_location = config.CHROME_BINARY_PATH
        service = Service(executable_path=config.CHROMEDRIVER_PATH)
        driver = webdriver.Chrome(service=service, options=options)

        if 'guba.eastmoney.com' in post_url:  # 这是绝大部分的普通帖子
            driver.get(post_url)
            date_str = driver.find_element(By.CSS_SELECTOR, 'div.newsauthor > div.author-info.cl > div.time').text
            self.year = int(self.remove_char(date_str)[:4])
            driver.quit()
        elif 'caifuhao.eastmoney.com' in post_url:  # 有些热榜帖子会占据第一位,对于这种情况要特殊处理
            driver.get(post_url)
            date_str = driver.find_element(By.CSS_SELECTOR, 'div.article.page-article > div.article-head > '
                                                            'div.article-meta > span.txt').text
            self.year = int(self.remove_char(date_str)[:4])
            driver.quit()
        else:
            self.year = datetime.now().year

    @staticmethod
    def judge_post_date(html):
        """判断帖子日期是否有效(东方财富有些显示不准确的日期)"""
        try:
            judge_element = html.find_element(By.CSS_SELECTOR, 'td:nth-child(3) > div > span')
            if judge_element.text == '问董秘':  # 存在“问董秘”标签,日期不可用
                return False
        except:
            return True

    def parse_post_date(self, html):
        """解析帖子发帖日期(月-日),并推算年份"""
        try:
            time_element = html.find_element(By.CSS_SELECTOR, 'div.update.pub_time')
            time_str = time_element.text
            month, day = map(int, time_str.split(' ')[0].split('-'))
        except Exception as e:  
            print('找不到帖子日期。', '\n', '{}'.format(e))
            return None, None

        if self.judge_post_date(html):
            if self.month < month == 12:
                self.year -= 1
            self.month = month

        if self.year is None:  # 如果年份尚未确定,通过详情页获取
            self.get_post_year(html)

        date = f'{self.year}{month:02d}{day:02d}'
        time = time_str.split(' ')[1]
        return date, time

    @staticmethod
    def parse_post_author(html):
        """解析帖子作者"""
        author_element = html.find_element(By.CSS_SELECTOR, 'td:nth-child(4) > div')
        return author_element.text

    def parse_post_info(self, html):
        """整合解析帖子所有信息,返回字典"""
        title = self.parse_post_title(html)
        view = self.parse_post_view(html)
        num = self.parse_comment_num(html)
        url = self.parse_post_url(html)
        date, time = self.parse_post_date(html)
        author = self.parse_post_author(html)
        m = re.search(r'news,[^,]+,(\d+)\.html', url)
        _id = int(m.group(1)) if m else int(hashlib.md5(url.encode()).hexdigest()[:16], 16)
        post_info = {
            '_id': _id,
            'post_title': title,
            'post_view': view,
            'comment_num': num,
            'post_url': url,
            'post_date': date,
            'post_time': time,
            'post_author': author
        }
        return post_info


def post_thread(stock_code, start_page, end_page, cutoff_date=None, target_start_date=None):
    """帖子爬取线程函数:stock_code为股票的代码,page为想要爬取的页面范围"""
    post_spider = PostSpider(stock_code)
    post_spider.spider_post_info(start_page, end_page, cutoff_date=cutoff_date, target_start_date=target_start_date)



if __name__ == "__main__":
    DB = "posts_test"

    code_1 = "000858"   
    start_1 = 1
    end_1 = 262

    # code_2 = ""
    # start_2 = 
    # end_2 = 

    # code_3 = ""
    # start_3 =   
    # end_3 = 

    
    # 爬取帖子信息
    thread1 = threading.Thread(target=post_thread, args=(code_1, start_1, end_1))  # 设置想要爬取的股票代码,开始与终止页数
    # thread2 = threading.Thread(target=post_thread, args=(code_2, start_2, end_2))  # 可同时进行多个线程
    # thread3 = threading.Thread(target=post_thread, args=(code_3, start_3, end_3))  # 可同时进行多个线程
 
    thread1.start()
    # thread2.start()
    # thread3.start()


    print(f"开始进行数据收集!")

