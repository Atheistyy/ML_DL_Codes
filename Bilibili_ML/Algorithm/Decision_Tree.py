# 信息熵： 度量样本纯度的一种指标
# 信息增益： 决策树划分依据之一
# 特征A对训练数据集D的信息增益g(D,A),定义为集合D的信息熵H(D)与特征A给定条件下D的信息条件熵H(D|A)之差，即公式为：
# g(D,A) = H(D) - H(D|A)

# 决策树API
# class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, max_depth=None,random_state=None)
# 决策树分类器
# criterion:默认是’gini’系数，也可以选择信息增益的熵’entropy’
# max_depth:树的深度大小
# random_state:随机数种子

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 决策树可视化
# 1、sklearn.tree.export_graphviz() 该函数能够导出DOT格式
# tree.export_graphviz(estimator,out_file='tree.dot’,feature_names=[‘’,’’])
# 2、可视化网站: webgraphviz.com

def decision_iris():
    # 获取数据集
    iris = load_iris()
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=6)
    # 决策树预估器
    estimator = DecisionTreeClassifier(criterion='entropy')
    estimator.fit(x_train,y_train)
    # 模型评估
    # 方法1: 直接对比真实值和预测值
    y_predict = estimator.predict(x_test)
    print('y_predict:\n', y_predict)
    print('直接对比真实值和预测值:\n', y_test == y_predict)
    # 方法2: 计算准确率
    accuracy = estimator.score(x_test, y_test)
    print('准确率为:\n', accuracy)
    #可视化决策树
    # export_graphviz(estimator, out_file='iris_tree.dot')
    plt.figure(figsize=(15, 9))
    plot_tree(estimator, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
    return None

decision_iris()

