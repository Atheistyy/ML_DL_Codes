# sklearn.cluster.KMeans(n_clusters=8,init=‘k-means++’)
# k-means聚类
# n_clusters:开始的聚类中心数量
# init:初始化方法，默认为'k-means ++’
# labels_:默认标记的类型，可以和真实值比较

# 评价指标: 轮廓系数
# sc_i = (b_i - a_i) / max(b_i, a_i)
# 如果b_i>>a_i:趋近于1效果越好， b_i<<a_i:趋近于-1，效果不好。
# 轮廓系数的值是介于 [-1,1] ，越趋近于1代表内聚度和分离度都相对较优。
# sklearn.metrics.silhouette_score(X, labels)
# 计算所有样本的平均轮廓系数
# X：特征值
# labels：被聚类标记的目标值



import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def KM():
    # 1、获取数据
    order_products = pd.read_csv("./instacart/order_products__prior.csv")
    products = pd.read_csv("./instacart/products.csv")
    orders = pd.read_csv("./instacart/orders.csv")
    aisles = pd.read_csv("./instacart/aisles.csv")
    # 2、合并表
    # order_products__prior.csv：订单与商品信息

    # 字段：order_id, product_id, add_to_cart_order, reordered
    # products.csv：商品信息
    # 字段：product_id, product_name, aisle_id, department_id
    # orders.csv：用户的订单信息
    # 字段：order_id,user_id,eval_set,order_number,….
    # aisles.csv：商品所属具体物品类别
    # 字段： aisle_id, aisle

    # 合并aisles和products aisle和product_id
    tab1 = pd.merge(aisles, products, on=["aisle_id", "aisle_id"])
    tab2 = pd.merge(tab1, order_products, on=["product_id", "product_id"])
    tab3 = pd.merge(tab2, orders, on=["order_id", "order_id"])
    # 3、找到user_id和aisle之间的关系
    table = pd.crosstab(tab3["user_id"], tab3["aisle"])
    data = table[:10000]
    # 4、PCA降维
    # 1）实例化一个转换器类
    transfer = PCA(n_components=0.95)

    # 2）调用fit_transform
    data_new = transfer.fit_transform(data)
    # 预估器流程
    estimator = KMeans(n_clusters=3)
    estimator.fit(data_new)
    y_predict = estimator.predict(data_new)
    # 模型评估-轮廓系数
    print(silhouette_score(data_new, y_predict))


KM()
