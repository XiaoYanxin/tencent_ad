import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# Dvalue = np.loadtxt(open("./D_values.csv", "rb"), delimiter = ",", skiprows = 0)
data = np.loadtxt(open("./mean_train_data.csv", "rb"), delimiter = ",", skiprows = 0).astype(int)
target = np.loadtxt(open("./mean_train_label.csv", "rb"), delimiter = ",", skiprows = 0)
# train_x1,test_x1,train_y1,test_y1 =train_test_split(data,target,test_size=0.2)
# train_x2,test_x2,train_y2,test_y2 =train_test_split(data,target,test_size=0.2)
# train_x3,test_x3,train_y3,test_y3 =train_test_split(data,target,test_size=0.2)
# train_x4,test_x4,train_y4,test_y4 =train_test_split(data,target,test_size=0.2)
# train_x5,test_x5,train_y5,test_y5 =train_test_split(data,target,test_size=0.2)
# X_train = np.vstack((train_x1,train_x2,train_x3,train_x4,train_x5))
# X_test = np.vstack((test_x1,test_x2,test_x3,test_x4,test_x5))
# y_train = np.hstack((train_y1,train_y2,train_y3,train_y4,train_y5))
# y_test = np.hstack((test_y1,test_y2,test_y3,test_y4,test_y5))
X_train,X_test,y_train,y_test =train_test_split(data,target,random_state=0,test_size=0.1)

#choose n_estimators
'''
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    # 'metric': 'rmse',
    'nthread': 8,
    'learning_rate': 0.1,
    'num_leaves': 50,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}

data_train = lgb.Dataset(X_train, y_train)
# cv_results = lgb.cv(params, data_train, num_boost_round=3000, nfold=3, stratified=False, shuffle=True, metrics='rmse',
#                     early_stopping_rounds=100, seed=0)
cv_results = lgb.cv(
    params, data_train, num_boost_round=10000, nfold=3, stratified=False, shuffle=True, metrics='rmse',
    early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)

print('best n_estimators:', len(cv_results['rmse-mean']))
print('best cv score:', pd.Series(cv_results['rmse-mean']).max())
# ###search parameters
# print('Start training...')
# # 创建模型，训练模型
# gbm = lgb.LGBMRegressor(objective='regression',num_leaves=100,learning_rate=0.1,n_estimators=500)
# gbm.fit(X_train, y_train,eval_set=[(X_test, y_test)],eval_metric='rmse',early_stopping_rounds=100)
#
# print('Start predicting...')
# # 测试机预测
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# # 模型评估
# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
#
# # feature importances
# print('Feature importances:', list(gbm.feature_importances_))
#
# # 网格搜索，参数优化
# estimator = lgb.LGBMRegressor(objective='regression', max_depth=7,learning_rate=0.1,n_estimators=500,num_leaves=100, metrics='rmse',
#                               subsample=0.8, colsample_bytree=0.8)
#
# param_grid = {
#     'n_estimators': []
# }
#
# gbm = GridSearchCV(estimator, param_grid,scoring='neg_mean_squared_error',n_jobs=3,verbose=1,return_train_score=True)
#
# gbm.fit(X_train, y_train)
# print((gbm.cv_results_))
# print('Best parameters found by grid search are:', gbm.best_params_,)
'''
# 创建成lgb特征的数据集格式
lgb_train = lgb.Dataset(X_train, y_train)  # 将数据保存到LightGBM二进制文件将使加载更快
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  # 创建验证数据

#将参数写成字典下形式
# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',  # 设置提升类型
#     'objective': 'regression',  # 目标函数
#     'metric': {'rmse','mae'},  # 评估函数 rmse优于l1
#     'num_leaves': 170,  # 叶子节点数170
#     'max_depth' : 8,
#     'lambda_l1' : 0.0001,
#     'lambda_l2' : 0.0,
#     'min_data_in_leaf': 20,
#     'max_bin in': 16,#16
#     'learning_rate': 0.001,  # 学习速率
#     'bagging_fraction':0.6,  # 建树的特征选择比例0.6
#     'feature_fraction': 0.65,  # 建树的样本采样比例0.65
#     'min_split_gain': 0.05,
#     'bagging_freq': 20,  # k 意味着每 k 次迭代执行bagging
#     'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
# }

params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression',  # 目标函数
    'metric': {'rmse','l1'},  # 评估函数 rmse优于l1
    'learning_rate': 0.001,  # 学习速率
    'num_leaves': 45,  # 叶子节点数170
    'max_depth' : 6,
    'lambda_l1' : 0.001,
    # 'lambda_l2' : 0.00001,
    'min_data_in_leaf': 25,
    # 'max_bin in': 15,
    'bagging_freq': 13,  # k 意味着每 k 次迭代执行bagging
    'bagging_fraction':0.75,
    'feature_fraction': 0.75,
    # 'min_split_gain': 0.4,
    'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}

print('Start training...')
# 训练 cv and train
gbm = lgb.train(params, lgb_train, num_boost_round=10000 ,valid_sets=lgb_eval,early_stopping_rounds=5000)

print('Save model...')

gbm.save_model('./models/model_w=0.0224_meandata.txt')  # 训练后保存模型到文件

print('Start predicting...')
# 预测数据集
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)  # 如果在训练期间启用了早期停止，可以通过best_iteration方式从最佳迭代中获得预测
# 评估模型
# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)  # 计算真实值和预测值之间的均方根误差
n = len(target)
s = abs(y_pred-y_test)
x = (y_pred+y_test)/2
SAMPE = (s/x).sum()/n
print('The SAMPE of prediction is:',SAMPE )

