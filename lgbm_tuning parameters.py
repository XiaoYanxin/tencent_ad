import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

data = np.loadtxt(open("./mean_train_data.csv", "rb"), delimiter = ",", skiprows = 0).astype(int)
target = np.loadtxt(open("./mean_train_label.csv", "rb"), delimiter = ",", skiprows = 0)
X_train,X_test,y_train,y_test =train_test_split(data,target,random_state=0,test_size=0.2)


#choose n_estimators
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'nthread': 8,
    'learning_rate': 0.1,
    'num_leaves': 100,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}

data_train = lgb.Dataset(X_train, y_train)
cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
                    early_stopping_rounds=100, seed=0)
# cv_results = lgb.cv(
#     params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
#     early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)

print('best n_estimators:', len(cv_results['rmse-mean']))
print('best cv score:', pd.Series(cv_results['rmse-mean']).max())
# RMSE:best n_estimators: 289 ,best cv score: 212.7707418568886
# MAE:best n_estimators: inf ,best cv score: 37.933071219886145


"""
# #chose max_depth and num_leaves
# params_test1 = {'max_depth':[6], 'num_leaves': [100]}
#
# gsearch1 = GridSearchCV(
#     estimator=lgb.LGBMRegressor(objective='regression', metrics='rmse', learning_rate=0.1,
#                                  n_estimators=289, max_depth=6, subsample=0.8, colsample_bytree=0.8),
#     param_grid=params_test1, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=4, return_train_score=True)#n_jobs  – Number of parallel threads.
# gsearch1.fit(X_train, y_train)
# gsearch1.grid_scores__, gsearch1.best_params_, gsearch1.best_score_

params_test1={
    'max_depth': [7],
    'num_leaves':[120]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=50,
                              learning_rate=0.1, n_estimators=50, max_depth=7,
                              metric='rmse', bagging_fraction = 0.8,feature_fraction = 0.8)
gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='neg_mean_squared_error',
                        cv=3, verbose=1, n_jobs=4, return_train_score=True)
gsearch1.fit(X_train, y_train)
gsearch1.grid_scores__, gsearch1.best_params_, gsearch1.best_score_

#choose min_data_in_leaf
# params_test2={'min_data_in_leaf':range(10,102,10)}
# gsearch2 = GridSearchCV(estimator = lgb.LGBMRegressor(objective='regression',metrics='rmse',learning_rate=0.1, n_estimators=289, max_depth=4, num_leaves=10,subsample=0.8, colsample_bytree=0.8,nthread=7),
#                        param_grid = params_test2, scoring='neg_mean_squared_error',cv=5,n_jobs=-1, return_train_score = True)
# gsearch2.fit(X_train,y_train)
# gsearch2.grid_scores__, gsearch2.best_params_, gsearch2.best_score_
#
# #choose colsample_bytree and subsample
# params_test3={'colsample_bytree': [0.6,0.7,0.8,0.9,1.0],
#               'subsample': [0.6,0.7,0.8,0.9,1.0]}
# gsearch3 = GridSearchCV(estimator = lgb.LGBMRegressor(objective='regression',metrics='rmse',learning_rate=0.1, n_estimators=289, max_depth=4, num_leaves=10,subsample=0.8, colsample_bytree=0.8,nthread=7),
#                        param_grid = params_test3, scoring='neg_mean_squared_error',cv=5,n_jobs=-1, return_train_score = True)
# gsearch3.fit(X_train,y_train)
# gsearch3.grid_scores__, gsearch3.best_params_, gsearch3.best_score_
#
# #choose lambda_l1 and lambda_l2
# params_test4={'lambda_l1': [1e-5,1e-3,1e-1,0.0,0.1,0.3],
#               'lambda_l2': [1e-5,1e-3,1e-1,0.0,0.1,0.3]}
# gsearch4 = GridSearchCV(estimator = lgb.LGBMRegressor(objective='regression',metrics='rmse',learning_rate=0.1, n_estimators=289, max_depth=4, num_leaves=10,subsample=0.8, colsample_bytree=0.8,nthread=7),
#                        param_grid = params_test4, scoring='neg_mean_squared_error',cv=5,n_jobs=-1, return_train_score = True)
# gsearch4.fit(X_train,y_train)
# gsearch4.grid_scores__, gsearch4.best_params_, gsearch4.best_score_
"""