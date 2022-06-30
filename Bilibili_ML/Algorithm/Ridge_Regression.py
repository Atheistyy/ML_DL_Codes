# 岭回归，其实也是一种线性回归。
# 只不过在算法建立回归方程时候，加上正则化的限制，从而达到解决过拟合的效果。
# API
# sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True,solver="auto", normalize=False)
# 具有l2正则化的线性回归
# alpha:正则化力度，也叫 λ
# λ取值：0~1 1~10
# solver:会根据数据自动选择优化方法
# SAG(随机平均梯度法):如果数据集、特征都比较大，选择该随机梯度下降优化
# normalize:数据是否进行标准化
# normalize=False:可以在fit之前调用preprocessing.StandardScaler标准化数据
# Ridge.coef_:回归权重
# Ridge.intercept_:回归偏置

# Ridge方法相当于SGDRegressor(penalty='l2', loss="squared_loss"),
# 只不过SGDRegressor实现了一个普通的随机梯度下降学习，推荐使用Ridge(实现了SAG)

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

def linear_3():
    # 获取数据
    boston = load_boston()
    # 划分数据集
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=22,test_size=0.2)
    # 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 预估器
    estimator = Ridge()
    estimator.fit(x_train,y_train)
    y_predict = estimator.predict(x_test)
    # 模型评估
    error = mean_squared_error(y_test, y_predict)
    # 模型数据
    print('岭回归-权重系数为:\n',estimator.coef_)
    print('岭回归-偏置为:\n',estimator.intercept_)
    print('岭回归-MSE:\n', error)
    return None

linear_3()
