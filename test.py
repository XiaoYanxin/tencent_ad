import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

data = np.loadtxt(open("./train_data.csv", "rb"), delimiter = ",", skiprows = 0).astype(int)
target = np.loadtxt(open("./train_label.csv", "rb"), delimiter = ",", skiprows = 0)
X_train,X_test,y_train,y_test =train_test_split(data,target,random_state=7,test_size=0.4)
df_features = pd.DataFrame(X_test,columns=['ad_id','Commodity_id','Commodity_type','Ad_Industry_Id','Ad_material_size',
                                           'ave_Delivery_time','age_feature','gender_feature','area_feature'])
df_labels = pd.DataFrame(y_test,columns=['cliscks-w*bid'])
df_features.to_csv('./compare/bagging_data.csv',index=False)
df_labels.to_csv('./compare/bagging_label.csv',index=False)