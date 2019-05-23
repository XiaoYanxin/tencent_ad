import numpy as np
import pandas as pd
import lightgbm as lgb
w = 0.0224
df = pd.read_csv('../TestB.csv')
# df = pd.read_csv('./compare/bagging_data.csv')
features = []
bids = []
for i in range(len(df)):
    l = []
    line = df.iloc[i]
    l.append(line['ad_id'])
    l.append(line['Commodity_id'])
    l.append(line['Commodity_type'])
    l.append(line['Ad_Industry_Id'])
    l.append(line['Ad_material_size'])
    l.append(line['ave_Delivery_time'])
    l.append(line['age_feature'])
    l.append(line['gender_feature'])
    l.append(line['area_feature'])
    bids.append(line['bid'])
    features.append(l)
bids = np.array(bids)
features = np.array(features, dtype=int)
model = lgb.Booster(model_file='./models/model_w=0.0224_meandata.txt')  #init model
pre = model.predict(features)
# result = pre + w*bids
# result = np.around(result,decimals=4)#保留4位小数
# N = len(result)
# print(N)
# result = np.where(result<0, 0, result)
# df_new = pd.DataFrame(result)
# df_new.index += 1

df_new = pd.DataFrame(pre)
df_new.to_csv('./data/submission_lgb_mean_TestB_nobid.csv',header=0,index=False)


###for bagging
# model1 = lgb.Booster(model_file='./models/model_w=0.0224_fulldata.txt')
# model2 = lgb.Booster(model_file='./models/model_w=0.0224_meandata.txt')
# pre1 = model1.predict(features)
# pre2 = model2.predict(features)
# df1 = pd.DataFrame(pre1,columns=['lgb_predict'])
# df2 = pd.DataFrame(pre2,columns=['lgb_predict'])
# df1.to_csv('./compare/lgb_fulldata_results.csv',index=False)
# df2.to_csv('./compare/lgb_meandata_results.csv',index=False)