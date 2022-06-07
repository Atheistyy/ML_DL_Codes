import pandas as pd
# 1. 读取数据
data = pd.read_csv('./facebook/train.csv')
# 2. 基本的数据处理
# 1) 缩小数据范围
data = data.query('x<2.5 & x>2 & y<1.5 & y>1')
# 2) 处理时间特征
time_value = pd.to_datetime(data['time'],unit = 's')
# time_value.values
date = pd.DatetimeIndex(time_value)
data['day'] = date.day
data['weekday'] = date.weekday
data['hour'] = date.hour
# 3) 过滤掉签到次数少的地点
# 每个地点的签到次数
place_count = data.groupby('place_id').count()['row_id']
# 只选择签到次数大于3的地点
place_count = place_count[place_count > 3]
# 筛选之后的数据集
data_final = data[data['place_id'].isin(place_count[place_count > 3].index.values)]

# 3. 筛选特征值和目标值
x = data_final[['x','y','accuracy','day','weekday','hour']]
y = data_final['place_id']

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
# 4. 数据集划分
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=6)
# 5. 数据标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# 6. KNN预估器
estimator = KNeighborsClassifier()
# 7. 加入网格搜索和交叉验证
param_dict = {'n_neighbors': [1,3,5,7,9,11]}
estimator = GridSearchCV(estimator, param_grid=param_dict, cv = 10)
estimator.fit(x_train, y_train)
# 8.模型评估
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
