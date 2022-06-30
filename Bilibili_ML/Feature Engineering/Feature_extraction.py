from sklearn.feature_extraction import DictVectorizer


# sklearn.feature_extraction.DictVectorizer(sparse=True,…)
# DictVectorizer.fit_transform(X) X:字典或者包含字典的迭代器返回值：返回sparse矩阵
# DictVectorizer.inverse_transform(X) X:array数组或者sparse矩阵 返回值:转换之前数据格式
# DictVectorizer.get_feature_names_out() 返回类别名称

def dict_demo():
    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]
    # 1、实例化一个转换器类
    transfer = DictVectorizer(sparse=False)  # sparse = Ture 返回的是稀疏矩阵
    # 2、调用fit_transform
    data = transfer.fit_transform(data)
    print("返回的结果:\n", data)
    # 打印特征名字
    print("特征名字：\n", transfer.get_feature_names_out())

    return None


dict_demo()
