# 交叉验证(cross validation)
# 为了让被评估的模型更加准确可信
# 训练集：训练集+验证集
# 测试集：测试集

# 超参数搜索-网格搜索(Grid Search)
# 通常情况下，有很多参数是需要手动指定的(如k-近邻算法中的K值)，
# 这种叫超参数。但是手动过程繁杂，所以需要对模型预设几种超参数组合。
# 每组超参数都采用交叉验证来进行评估。最后选出最优参数组合建立模型。

# 模型选择与调优
# sklearn.model_selection.GridSearchCV(estimator, param_grid=None,cv=None)
# 对估计器的指定参数值进行详尽搜索
# estimator：估计器对象
# param_grid：估计器参数(dict){“n_neighbors”:[1,3,5]}
# cv：指定几折交叉验证
# fit：输入训练数据
# score：准确率
# 结果分析：
# bestscore:在交叉验证中验证的最好结果_
# bestestimator：最好的参数模型
# cvresults:每次交叉验证后的验证集准确率结果和训练集准确率结果

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def knn_iris_gscv():
    # 1. 获取数据集
    iris = load_iris()
    # 2. 划分数据集
    x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target, random_state = 6) # or test_size = 0.2
    # 3. 特征工程: 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4. KNN预估器
    estimator = KNeighborsClassifier()


    # 加入网格搜索和交叉验证
    param_dict = {'n_neighbors': [1,3,5,7,9,11]}  # 超参数 k 的取值
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv = 10)
    # cv 表示每取一个 k 值，进行几折的交叉验证，一般取10
    estimator.fit(x_train, y_train)

    # 5. 模型评估
    # 方法1: 直接对比真实值和预测值
    y_predict = estimator.predict(x_test)
    print('y_predict:\n',y_predict)
    print('直接对比真实值和预测值:\n',y_test == y_predict)
    # 方法2: 计算准确率
    accuracy = estimator.score(x_test, y_test)
    print('准确率为:\n', accuracy)

    print('最佳参数:\n', estimator.best_params_)
    print('最佳结果:\n',estimator.best_score_ )  # 此处的最佳结果指的是 交叉验证集中的准确率
    print('最佳预估器:\n',estimator.best_estimator_)
    print('交叉验证结果:\n',estimator.cv_results_)
    return None

knn_iris_gscv()
