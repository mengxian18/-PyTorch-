import pandas as pd

# 读取数据
file_path = r'D:\code\ai+\test\data\amzn-anon-access-samples\amzn-anon-access-samples-history-2.0.csv'
data = pd.read_csv(file_path)

# 定义日期列名
date_columns = ['REQUEST_DATE', 'AUTHORIZATION_DATE']

# 统计每列的异常数据数量
for column in date_columns:
    try:
        # 尝试转换日期列，无法转换的会变成 NaT
        converted = pd.to_datetime(data[column], errors='coerce')
        # 统计 NaT 的数量，即异常数据数量
        abnormal_count = converted.isna().sum()
        print(f'{column} 列中的异常数据数量为: {abnormal_count}')
    except KeyError:
        print(f'数据集中不存在 {column} 列。')