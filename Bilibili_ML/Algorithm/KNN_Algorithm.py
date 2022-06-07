# K-近邻算法(KNN)
# 如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别。
# API
# sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,algorithm='auto')
# n_neighbors：int,可选（默认= 5），k_neighbors查询默认使用的邻居数
# algorithm：{‘auto’，‘ball_tree’，‘kd_tree’，‘brute’}，可选用于计算最近邻居的算法：
# ‘ball_tree’将会使用 BallTree，
# ‘kd_tree’将使用 KDTree。
# ‘auto’将尝试根据传递给fit方法的值来决定最合适的算法。 (不同实现方式影响效率)

# 以鸢尾花为例
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def knn_iris():
    # 1. 获取数据集
    iris = load_iris()
    # 2. 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)  # or test_size = 0.2
    # 3. 特征工程: 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4. KNN预估器
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    # 5. 模型评估
    # 方法1: 直接对比真实值和预测值
    y_predict = estimator.predict(x_test)
    print('y_predict:\n', y_predict)
    print('直接对比真实值和预测值:\n', y_test == y_predict)
    # 方法2: 计算准确率
    accuracy = estimator.score(x_test, y_test)
    print('准确率为:\n', accuracy)
    return None


knn_iris()

# 优点：
# 简单，易于理解，易于实现，无需训练
# 缺点：
# 懒惰算法，对测试样本分类时的计算量大，内存开销大
# 必须指定K值，K值选择不当则分类精度不能保证
# 使用场景：小数据场景，几千～几万样本，具体场景具体业务去测试
