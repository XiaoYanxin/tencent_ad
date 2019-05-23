import numpy as np
import pandas as pd

f1 = np.array(pd.read_csv('./data/submission_lgb_mean_TestB_nobid.csv',header=None))
print(f1.shape)
f2 = np.array(pd.read_csv('./data/submission_lgb_full_TestB_nobid.csv',header=None))
print(f2.shape)
f3 = np.array(pd.read_csv('./data/submission_nn_mean_TestB_nobid.csv'))
print(f3.shape)
f4 = np.array(pd.read_csv('./data/submission_nn_full_TestB_nobid.csv'))
print(f4.shape)
f5 = np.array(pd.read_csv('./data/submission_rf_mean_TestB_nobid.csv'))
print(f5.shape)
f6 = np.array(pd.read_csv('./data/submission_rf_full_TestB_nobid.csv'))
print(f6.shape)
# w = np.array(pd.read_csv('./data/W.csv'))
w = 0.0224
w1,w2,w3,w4,w5,w6 = 0.434,0.186,0.126,0.054,0.14,0.06
# w1,w2,w3,w4,w5,w6 = 0.56,0.24,0.0,0.0,0.14,0.06
pre = w1*f1 + w2*f2 + w3*f3 + w4*f4 + w5*f5 + w6*f6
# print(pre.shape)
# bids = np.array(df['bid'])
bids = np.array(pd.read_csv('./data/testB_bid.csv'))
print(bids.shape)
result = pre + w*bids
# print(result)
print(result.shape)
result = np.around(result,decimals=4)#保留4位小数
N = len(result)
print(N)
result = np.where(result<0, 0, result)
df_new = pd.DataFrame(result)
df_new.index += 1
df_new.to_csv('./data/old_submission.csv',index=False,header=0)