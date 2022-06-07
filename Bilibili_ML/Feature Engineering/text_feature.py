from sklearn.feature_extraction.text import CountVectorizer
import jieba


def text_count_demo():
    data = ["life is short,i like like python", "life is too long,i dislike python"]
    # 1、实例化一个转换器类
    transfer = CountVectorizer(stop_words=['is', 'too'])  # stop_words，添加停用词
    # 2、调用fit_transform
    data = transfer.fit_transform(data)
    print("文本特征抽取的结果：\n", data.toarray())
    print("返回特征名字：\n", transfer.get_feature_names_out())

    return None


text_count_demo()


# 停用词表


# 中文分词
# jieba.cut()
# 返回词语组成的生成器
def cut_word(text):
    # 进行中文分词
    split_text = " ".join(jieba.cut(text))
    print(split_text)
    return (split_text)


cut_word("我的宝贝王小懒早上买了一个卷饼")


def text_chinese_count_demo():
    """
    对中文进行特征抽取
    :return: None
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    # 将原始数据转换成分好词的形式
    data_new = [cut_word(x) for x in data]
    # 1、实例化一个转换器类
    transfer = CountVectorizer()
    # 2、调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print("文本特征抽取的结果：\n", data_final.toarray())
    print("返回特征名字：\n", transfer.get_feature_names_out())

    return None


text_chinese_count_demo()

# Tf-idf文本特征提取

# TF-IDF的主要思想是：如果某个词或短语在一篇文章中出现的概率高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。
# TF-IDF作用：用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。

# 词频（term frequency，tf）指的是某一个给定的词语在该文件中出现的频率
# 逆向文档频率（inverse document frequency，idf）是一个词语普遍重要性的度量。
# 某一特定词语的idf，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取以10为底的对数得到
# 最终得出结果可以理解为重要程度。
# 假如一篇文件的总词语数是100个，而词语"非常"出现了5次，
# 那么"非常"一词在该文件中的词频就是5/100=0.05。
# 而计算文件频率（IDF）的方法是以文件集的文件总数，
# 除以出现"非常"一词的文件数。
# 所以，如果"非常"一词在1,000份文件出现过，
# 而文件总数是10,000,000份的话，其逆向文件频率就是lg（10,000,000 / 1,0000）=3。
# 最后"非常"对于这篇文档的tf-idf的分数为0.05 * 3=0.15
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_count_demo():
    """
    对中文进行特征抽取
    :return: None
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    # 将原始数据转换成分好词的形式
    data_new = [cut_word(x) for x in data]
    # 1、实例化一个转换器类
    transfer = TfidfVectorizer()
    # 2、调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print("文本特征抽取的结果：\n", data_final.toarray())
    print("返回特征名字：\n", transfer.get_feature_names_out())

    return None


tfidf_count_demo()
