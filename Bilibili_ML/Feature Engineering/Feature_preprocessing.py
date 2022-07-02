# 特征预处理: 通过一些转换函数将特征数据转换成更加适合算法模型的特征数据过程
# 包含内容 —— 数值型数据的无量纲化：
# 归一化和标准化
# 为什么:
# 特征的单位或者大小相差较大，或者某特征的方差相比其他的特征要大出几个数量级，
# 容易影响（支配）目标结果，使得一些算法无法学习到其它的特征

# 归一化定义: 通过对原始数据进行变换把数据映射到(默认为[0,1])之间
# 计算公式: X' = (x - x_max) / (x_max - x_min)
# X" = X'*(mx - mi) + mi   mx,mi为指定区间值，默认为1和0

# API
# sklearn.preprocessing.MinMaxScaler (feature_range=(0,1)… )
# MinMaxScalar.fit_transform(X)
# X:numpy array格式的数据[n_samples,n_features]
# 返回值：转换后的形状相同的array

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def minmax_demo():
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]  # 索引方式，表示所有行都要，并取前三列
    print(data)
    # 1、实例化一个转换器类
    transfer = MinMaxScaler(feature_range=(2, 3))  # 输出范围在2-3，也可以设置为0-1,etc.
    # 2、调用fit_transform
    data = transfer.fit_transform(data[['milage', 'Liters', 'Consumtime']])
    print("最小值最大值归一化处理的结果：\n", data)

    return None


minmax_demo()

# 注意最大值最小值是变化的，另外，最大值与最小值非常容易受异常点影响，
# 所以这种方法鲁棒性较差，只适合传统精确小数据场景。


# 标准化定义: 通过对原始数据进行变换把数据变换到均值为0,标准差为1范围内
# 计算公式: X' = (x - mean) / std , mean为均值，std为标准差
# 对于归一化来说：如果出现异常点，影响了最大值和最小值，那么结果显然会发生改变
# 对于标准化来说：如果出现异常点，由于具有一定数据量，
# 少量的异常点对于平均值的影响并不大，从而方差改变较小。

# API
# sklearn.preprocessing.StandardScaler( )
# 处理之后每列来说所有数据都聚集在均值0附近标准差差为1
# StandardScaler.fit_transform(X)
# X:numpy array格式的数据[n_samples,n_features]
# 返回值：转换后的形状相同的array

import pandas as pd
from sklearn.preprocessing import StandardScaler


def stand_demo():
    data = pd.read_csv("dating.txt")
    print(data)
    # 1、实例化一个转换器类
    transfer = StandardScaler()
    # 2、调用fit_transform
    data = transfer.fit_transform(data[['milage', 'Liters', 'Consumtime']])
    print("标准化的结果:\n", data)
    print("每一列特征的平均值：\n", transfer.mean_)
    print("每一列特征的方差：\n", transfer.var_)

    return None


stand_demo()

# 标准化 - 在已有样本足够多的情况下比较稳定，适合现代嘈杂大数据场景。
