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
# 决策树预估器
estimator = DecisionTreeClassifier(criterion='entropy')
estimator.fit(x_train, y_train)
# 模型评估
# 方法1: 直接对比真实值和预测值
y_predict = estimator.predict(x_test)
print('y_predict:\n', y_predict)
print('直接对比真实值和预测值:\n', y_test == y_predict)
# 方法2: 计算准确率
accuracy = estimator.score(x_test, y_test)
print('准确率为:\n', accuracy)


# 可视化
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(15, 9))
plot_tree(estimator, filled=True, feature_names=transfer.get_feature_names_out(), class_names='Survived')

