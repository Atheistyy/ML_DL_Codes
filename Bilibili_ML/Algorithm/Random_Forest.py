# 集成学习通过建立几个模型组合的来解决单一预测问题。
# 它的工作原理是生成多个分类器/模型，各自独立地学习和作出预测。
# 这些预测最后结合成组合预测，因此优于任何一个单分类的做出预测。

# 在机器学习中，随机森林是一个包含多个决策树的分类器，
# 并且其输出的类别是由个别树输出的类别的众数而定。

# API
# class sklearn.ensemble.RandomForestClassifier(...) ,如下所示
# (n_estimators=10, criterion=’gini’, max_depth=None, bootstrap=True, random_state=None, min_samples_split=2)
# 随机森林分类器
# n_estimators：integer，optional（default = 10）森林里的树木数量120,200,300,500,800,1200
# criteria：string，可选（default =“gini”）分割特征的测量方法
# max_depth：integer或None，可选（默认=无）树的最大深度 5,8,15,25,30
# max_features="auto”,每个决策树的最大特征数量
# If "auto", then max_features=sqrt(n_features).
# If "sqrt", then max_features=sqrt(n_features) (same as "auto").
# If "log2", then max_features=log2(n_features).
# If None, then max_features=n_features.
# bootstrap：boolean，optional（default = True）是否在构建树时使用放回抽样
# min_samples_split:节点划分最少样本数
# min_samples_leaf:叶子节点的最小样本数
# 超参数：n_estimator, max_depth, min_samples_split,min_samples_leaf



import pandas as pd
# 读取数据
titanic = pd.read_csv('./titanic/train.csv')
x = titanic[['Pclass', 'Age', 'Sex']]
y = titanic['Survived']
# print(x.head())

# 数据处理
# 缺失值处理
x['Age'].fillna(x['Age'].mean(), inplace=True)
# 对于x转换成字典数据x.to_dict(orient="records")
# [{"pclass": "1st", "age": 29.00, "sex": "female"}, {}]
x = x.to_dict(orient='records')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)
# 字典特征抽取
transfer = DictVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 随机森林去进行预测
rf = RandomForestClassifier()
param = {"n_estimators": [120,200,300,500,800,1200], "max_depth": [5, 8, 15, 25, 30]}
# 超参数调优
gc = GridSearchCV(rf, param_grid=param, cv=3)
gc.fit(x_train, y_train)
print("随机森林预测的准确率为：", gc.score(x_test, y_test))

print('最佳参数:\n', gc.best_params_)
print('最佳结果:\n', gc.best_score_)  # 此处的最佳结果指的是 交叉验证集中的准确率
print('最佳预估器:\n', gc.best_estimator_)
print('交叉验证结果:\n', gc.cv_results_)

