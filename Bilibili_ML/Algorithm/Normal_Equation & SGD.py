# 线性回归API
# sklearn.linear_model.LinearRegression(fit_intercept=True)
# 通过正规方程优化
# fit_intercept：是否计算偏置
# LinearRegression.coef_：回归系数
# LinearRegression.intercept_：偏置
# sklearn.linear_model.SGDRegressor
# (loss="squared_loss", fit_intercept=True, learning_rate ='invscaling', eta0=0.01)
# SGDRegressor类实现了随机梯度下降学习，它支持不同的loss函数和正则化惩罚项来拟合线性回归模型。
# loss:损失类型
# loss=”squared_loss”: 普通最小二乘法
# fit_intercept：是否计算偏置
# learning_rate : string, optional
# 学习率填充
# 'constant': eta = eta0
# 'optimal': eta = 1.0 / (alpha * (t + t0)) [default]
# 'invscaling': eta = eta0 / pow(t, power_t)
# power_t=0.25:存在父类当中
# 对于一个常数值的学习率来说，可以使用learning_rate=’constant’ ，并使用eta0来指定学习率。
# SGDRegressor.coef_：回归系数
# SGDRegressor.intercept_：偏置

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error

boston = load_boston()
print('特征数量:\n', boston.data.shape)


# 基于正规方程的线性模型
def linear_1():
    # 获取数据
    boston = load_boston()
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22, test_size=0.2)
    # 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 预估器
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    # 模型评估
    error = mean_squared_error(y_test, y_predict)
    # 模型数据
    print('正规方程-权重系数为:\n', estimator.coef_)
    print('正规方程-偏置为:\n', estimator.intercept_)
    print('正规方程-MSE:\n', error)
    return None


# 基于梯度下降的线性模型
def linear_2():
    # 获取数据
    boston = load_boston()
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22, test_size=0.2)
    # 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 预估器
    estimator = SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=10000)
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    # 模型评估
    error = mean_squared_error(y_test, y_predict)
    # 模型数据
    print('梯度下降-权重系数为:\n', estimator.coef_)
    print('梯度下降-偏置为:\n', estimator.intercept_)
    print('梯度下降-MSE:\n', error)
    return None


linear_1()
linear_2()

# 选择：
# 小规模数据：
# LinearRegression(不能解决拟合问题)
# 岭回归
# 大规模数据：SGDRegressor
