# 朴素: 特征和特征之间是独立的
# P(C/F1,F2...) = (P(F1,F2.../C) * P(C)) / P(F1,F2,...)
# 拉普拉斯平滑系数: 防止计算出的分类概率为0
# P(F1/C) = (Ni + a) / (N + am)
# a 为指定的系数一般为1，m为训练文档中统计出的特征词个数
# API
# sklearn.naive_bayes.MultinomialNB(alpha = 1.0)
# 朴素贝叶斯分类
# alpha：拉普拉斯平滑系数
# 优点：
# 朴素贝叶斯模型发源于古典数学理论，有稳定的分类效率。
# 对缺失数据不太敏感，算法也比较简单，常用于文本分类。
# 分类准确度高，速度快
# 缺点：
# 由于使用了样本属性独立性的假设，所以如果特征属性有关联时其效果不好

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def nb_news():
    # 获取数据
    news = fetch_20newsgroups(subset="all")
    # 划分数据集
    x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.2,random_state=6)
    # 文本特征抽取 - tfidf
    tf = TfidfVectorizer()
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)
    # 朴素贝叶斯算法评估器预估模型
    estimator = MultinomialNB()
    estimator.fit(x_train,y_train)
    # 模型评估
    # 方法1: 直接对比真实值和预测值
    y_predict = estimator.predict(x_test)
    print('y_predict:\n', y_predict)
    print('直接对比真实值和预测值:\n', y_test == y_predict)
    # 方法2: 计算准确率
    accuracy = estimator.score(x_test, y_test)
    print('准确率为:\n', accuracy)
    return None

nb_news()

