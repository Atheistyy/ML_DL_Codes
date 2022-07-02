# sklearn.datasets
# 加载获取流行数据集
# datasets.load_*()
# 获取小规模数据集，数据包含在datasets里
# datasets.fetch_*(data_home=None)
# 获取大规模数据集，需要从网络上下载，
# 函数的第一个参数是data_home，表示数据集下载的目录,默认是 ~/scikit_learn_data/

# 返回值是一个继承自字典的Bench
# data：特征数据数组，是 [n_samples * n_features] 的二维 numpy.ndarray 数组
# target：标签数组，是 n_samples 的一维 numpy.ndarray 数组
# DESCR：数据描述
# feature_names：特征名,新闻数据，手写数字、回归数据集没有
# target_names：标签名

import sklearn
from sklearn.datasets import load_iris


# 1、获取鸢尾花数据集
def datasets_demo():
    iris = load_iris()  # 获取数据集
    print("鸢尾花数据集的返回值：\n", iris)
    print("鸢尾花的特征值:\n", iris.data, iris.data.shape)
    print("鸢尾花的目标值：\n", iris.target)
    print("鸢尾花特征的名字：\n", iris.feature_names)
    print("鸢尾花目标值的名字：\n", iris.target_names)
    print("鸢尾花的描述：\n", iris.DESCR)
    return None


datasets_demo()

# 2、对鸢尾花数据集进行分割
from sklearn.model_selection import train_test_split

# sklearn.model_selection.train_test_split(arrays, *options)
# x 数据集的特征值
# y 数据集的标签值
# test_size 测试集的大小，一般为float
# random_state 随机数种子,不同的种子会造成不同的随机采样结果。相同的种子采样结果相同。
# return 训练集特征值，测试集特征值，训练标签，测试标签(默认随机取)
iris = load_iris()
# 训练集的特征值x_train 测试集的特征值x_test 训练集的目标值y_train 测试集的目标值y_test
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
print("x_train:\n", x_train.shape)
