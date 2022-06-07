# 转换器 （特征工程的父类）
# 1、实例化 (实例化的是一个转换器类(Transformer))
# 2、调用fit_transform(对于文档建立分类词频矩阵，不能同时调用)
# 转换器调用形式:
# fit_transform
# fit
# transform
# 以标准化为例 fit_transform()
# fit() —— 计算每一列的平均值和标准差
# transform() —— (x - mean) / std 进行最终的转换

# 估计器(sklearn机器学习算法的实现)
# 在sklearn中，估计器(estimator)是一个重要的角色，是一类实现了算法的API
# 1、用于分类的估计器：
# sklearn.neighbors k-近邻算法
# sklearn.naive_bayes 贝叶斯
# sklearn.linear_model.LogisticRegression 逻辑回归
# sklearn.tree 决策树与随机森林
# 2、用于回归的估计器：
# sklearn.linear_model.LinearRegression 线性回归
# sklearn.linear_model.Ridge 岭回归
# 3、用于无监督学习的估计器
# sklearn.cluster.KMeans 聚类

# 工作流程:
# 1. 实例化一个estimator
# 2. estimator.fit(x_train, y_train)  计算 —— 调用完毕，模型生成
# 3. 模型评估:
# 直接比对真实值和预测值 —— y_predict = estimator.predict(x_test) , y_test == y_predict
# 计算准确率 —— accuracy = estimator.score(x_test, y_test)
