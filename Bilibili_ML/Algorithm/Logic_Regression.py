# sklearn.linear_model.LogisticRegression(solver='liblinear', penalty=‘l2’, C = 1.0)
# solver:优化求解方式（默认开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数）
# sag：根据数据集自动选择，随机平均梯度下降
# penalty：正则化的种类
# C：正则化力度

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

def logistic():
    # 1. 读取数据，处理缺失值以及标准化
    path = './cancer/breast-cancer-wisconsin.data'
    column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv(path, names=column_name)
    # 2. 缺失值处理
    # 1）替换
    data = data.replace(to_replace='?', value=np.nan)
    # 2）删除缺失样本(处理NAN的方式)
    data.dropna(inplace=True)
    data.isnull().any()  # 检查是否还存在缺失值
    # 3. 划分数据集
    x = data.iloc[:, 1:-1]
    y = data['Class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6)
    # 4. 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 5. 预估器
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    # 逻辑回归的模型参数： 权重和偏置
    print('权重为:\n', estimator.coef_)
    print('偏置为:\n', estimator.intercept_)
    # 6. 模型评估
    error = mean_squared_error(y_test, y_predict)
    print('MSE:\n', error)
    # 直接对比真实值和预测值
    y_predict = estimator.predict(x_test)
    print('y_predict:\n', y_predict)
    print('直接对比真实值和预测值:\n', y_test == y_predict)
    # 计算准确率
    accuracy = estimator.score(x_test, y_test)
    print('准确率为:\n', accuracy)
    # 查看精确率、召回率、F1-score
    report = classification_report(y_test, y_predict, labels=[2, 4], target_names=["良性", "恶性"])
    print(report)
    # y_true：每个样本的真实类别，必须为0(反例),1(正例)标记
    # 将y_test 转换成 0 1
    y_true = np.where(y_test > 3, 1, 0)
    score = roc_auc_score(y_true, y_predict)
    print(score)
    return None


logistic()
