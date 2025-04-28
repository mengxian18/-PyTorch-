import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from causalinference import CausalModel

# 读取数据
data = pd.read_csv(r'D:\code\ai+\test\data\amzn-anon-access-samples\amzn-anon-access-samples-history-2.0.csv')

# 数据预处理
# 尝试转换日期列并删除异常数据
def convert_date_column(column):
    converted = pd.to_datetime(column, errors='coerce')
    # 找出无法转换的数据
    abnormal_mask = converted.isna()
    abnormal_count = abnormal_mask.sum()
    print(f"发现 {abnormal_count} 条异常数据，已删除。")
    # 删除异常数据
    valid_column = column[~abnormal_mask]
    return pd.to_datetime(valid_column, format='mixed').astype(np.int64)

data['REQUEST_DATE'] = convert_date_column(data['REQUEST_DATE'])
data['AUTHORIZATION_DATE'] = convert_date_column(data['AUTHORIZATION_DATE'])

# 删除包含 NaN 的行
data = data.dropna(subset=['REQUEST_DATE', 'AUTHORIZATION_DATE'])
# 因果分析
# 假设 ACTION 为处理变量（treatment），REQUEST_DATE 转换为时间戳作为协变量
# 这里简单将 ACTION 分为 add_access 和 remove_access 两种处理组
data['treatment'] = (data['ACTION'] == 'add_access').astype(int)
data['timestamp'] = data['REQUEST_DATE'].astype(np.int64) // 10**9

# 选择因果分析所需的变量
treatment = data['treatment'].values
outcome = data['AUTHORIZATION_DATE'].astype(np.int64) // 10**9
covariates = data[['timestamp']].values

# 创建因果模型
causal = CausalModel(outcome, treatment, covariates)

# 估计因果效应
try:
    causal.est_via_ols()
except ValueError as e:
    if 'Multi-dimensional indexing' in str(e):
        # 手动修改 causalinference 库中的相关代码
        # 假设问题出在 calc_cov 函数中的 u[:, None]*Z 操作
        from causalinference.estimators.ols import calc_cov
        def new_calc_cov(Z, u):
            u = u.values if hasattr(u, 'values') else u  # 转换为 numpy 数组
            Z = Z.values if hasattr(Z, 'values') else Z  # 转换为 numpy 数组
            A = np.linalg.inv(np.dot(Z.T, Z))
            B = np.dot(u[:, None]*Z, A)
            return np.dot(B.T, B)
        # 替换原有的 calc_cov 函数
        from causalinference.estimators import ols
        ols.calc_cov = new_calc_cov
        causal.est_via_ols()
    else:
        raise e

# 输出因果分析结果
print("因果分析结果：")
print(causal.estimates)