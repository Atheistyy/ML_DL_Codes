# 案例：探究用户对物品类别的喜好细分降维

# 数据如下：
# order_products__prior.csv：订单与商品信息
# 字段：order_id, product_id, add_to_cart_order, reordered
# products.csv：商品信息
# 字段：product_id, product_name, aisle_id, department_id
# orders.csv：用户的订单信息
# 字段：order_id,user_id,eval_set,order_number,….
# aisles.csv：商品所属具体物品类别
# 字段： aisle_id, aisle

# 合并表，使得user_id与aisle在一张表当中
# 进行交叉表变换
# 进行降维

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

products = pd.read_csv("./instacart/products.csv")
order_products = pd.read_csv("./instacart/order_products__prior.csv")
orders = pd.read_csv("./instacart/orders.csv")
aisles = pd.read_csv("./instacart/aisles.csv")

# 2、合并表，将user_id和aisle放在一张表上
# 合并 aisle 和 product_id, 因为其都含有aisles_id这一项
tab1 = pd.merge(aisles, products, on=['aisle_id', 'aisle_id'])

tab2 = pd.merge(tab1, order_products, on=['product_id', 'product_id'])

tab3 = pd.merge(tab2, orders, on=['order_id', 'order_id'])

# 3、交叉表处理，把user_id和aisle进行分组
table = pd.crosstab(tab3["user_id"], tab3["aisle"])
print(table.shape)
# 4、主成分分析的方法进行降维
# 1）实例化一个转换器类PCA
transfer = PCA(n_components=0.95)
# 2）fit_transform
data = transfer.fit_transform(table)
print(data.shape)
