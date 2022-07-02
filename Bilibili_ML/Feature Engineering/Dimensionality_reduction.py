# 降维: 降维是指在某些限定条件下，降低随机变量(特征)个数，得到一组“不相关”主变量的过程

# 降维的两种方式
# 特征选择
# 主成分分析（可以理解一种特征提取的方式）

# 特征选择定义: 数据中包含冗余或无关变量（或称特征、属性、指标等），旨在从原有特征中找出主要特征
# 方法如下
# Filter(过滤式)：主要探究特征本身特点、特征与特征和目标值之间关联
# 方差选择法：低方差特征过滤相关系数
# Embedded (嵌入式)：算法自动选择特征（特征与目标值之间的关联）
# 决策树:信息熵、信息增益
# 正则化：L1、L2
# 深度学习：卷积等


# 一. 过滤式: 低方差特征过滤
# 删除低方差的一些特征，再结合方差的大小来考虑这个方式的角度。
# 特征方差小：某个特征大多样本的值比较相近
# 特征方差大：某个特征很多样本的值都有差别

# API
# sklearn.feature_selection.VarianceThreshold(threshold = 0.0)
# 删除所有低方差特征
# Variance.fit_transform(X)
# X:numpy array格式的数据[n_samples,n_features]
# 返回值：训练集差异低于threshold的特征将被删除。默认值是保留所有非零方差特征，即删除所有样本中具有相同值的特征。
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def variance_demo():
    # 1. 获取数据
    data = pd.read_csv('factor_returns.csv')
    # 我们对某些股票的指标特征之间进行一个筛选，
    # 数据在"factor_regression_data/factor_returns.csv"文件当中,
    # 除去'index,'date','return'列不考虑（这些类型不匹配，也不是所需要指标）

    data = data.iloc[:, 1:-2]  # 表示所有行都要，列只要 pe_ratio - total_expense
    print(data)
    # 2. 实例化一个转化器类
    transfer = VarianceThreshold(threshold=10)

    # 3. 调用fit.transform
    data_new = transfer.fit_transform(data)
    print('data_new:\n', data_new, data_new.shape)
    return None


# variance_demo()


# 相关系数
# 皮尔逊相关系数(Pearson Correlation Coefficient)
# 反映变量之间相关关系密切程度的统计指标
# 相关系数的值介于–1与+1之间，即–1≤ r ≤+1。其性质如下：
# 当r>0时，表示两变量正相关，r<0时，两变量为负相关
# 当|r|=1时，表示两变量为完全相关，当r=0时，表示两变量间无相关关系
# 当0<|r|<1时，表示两变量存在一定程度的相关。且|r|越接近1，两变量间线性关系越密切；|r|越接近于0，表示两变量的线性相关越弱
# 一般可按三级划分：|r|<0.4为低度相关；0.4≤|r|<0.7为显著性相关；0.7≤|r|<1为高度线性相关

# API
# from scipy.stats import pearsonr
# x : (N,) array_like
# y : (N,) array_like Returns: (Pearson’s correlation coefficient, p-value)

from scipy.stats import pearsonr
import matplotlib.pyplot as plt

data = pd.read_csv('factor_returns.csv')
r1 = pearsonr(data['pe_ratio'], data['pb_ratio'])
print('pe_ratio & pb_ratio 相关系数(第一个值表示相关系数):\n', r1)

r2 = pearsonr(data['revenue'], data['total_expense'])
print('revenue & total_expense 相关系数:\n', r2)

plt.figure(figsize=(5, 4), dpi=100)
plt.scatter(data['revenue'], data['total_expense'])
plt.show()

# 如果两个特征之间相关性很高
# 解决办法:
# 1. 两者取其一
# 2. 加权求和
# 3. 主成分分析（自动处理）


# 二. 主成分分析(PCA)
# 定义：高维数据转化为低维数据的过程，在此过程中可能会舍弃原有数据、创造新的变量
# 作用：是数据维数压缩，尽可能降低原数据的维数（复杂度），损失少量信息
# 应用：回归分析或者聚类分析当中

# API
# sklearn.decomposition.PCA(n_components=None)
# 将数据分解为较低维数空间
# n_components:
# 小数：表示保留百分之多少的信息
# 整数：减少到多少特征
# PCA.fit_transform(X) X:numpy array格式的数据[n_samples,n_features]
# 返回值：转换后指定维度的array
from sklearn.decomposition import PCA


def pca_demo():
    """
    对数据进行PCA降维
    :return: None
    """
    # 一个简单例子
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]
    # 1、实例化PCA, 小数 —— 保留多少信息
    transfer = PCA(n_components=0.9)
    # 2、调用fit_transform
    data1 = transfer.fit_transform(data)
    print("保留90%的信息，降维结果为：\n", data1)

    # 1、实例化PCA, 整数 —— 指定降维到的维数
    transfer2 = PCA(n_components=3)
    # 2、调用fit_transform
    data2 = transfer2.fit_transform(data)
    print("降维到3维的结果：\n", data2)
    return None


pca_demo()
