import pandas as pd
import numpy as np
from collections import defaultdict

df = pd.read_csv('../new.csv')
N = len(df)
dic = defaultdict(list)
delete = []
mem = []
for i in range(N):
    l = df.iloc[i]
    dic[str(l['ad_id'])+str(l['ave_Delivery_time'])+str(l['age_feature'])+str(l['gender_feature'])+str(l['area_feature'])].append((int(l['ad_bid']),int(l['num_click']),i))
    # dic[str(l['ad_id'])+str(l['ave_Delivery_time'])+str('Chose_People')].append((int(l['ad_bid']),int(l['num_click']),i))
print(len(dic))
print(dic)
for k in list(dic.keys()):
    l1 = dic[k]
    l2 = dic[k]
    l1.sort(key = lambda x:x[0])
    l2.sort(key = lambda x:x[1])
    if l1 == l2:
        continue
        if len(l1)>1:
            mem.append(l1)
    else:
        print('found outlier')
        for t in l1:
            delete.append(t[-1])
print("delete %d samples"%(len(delete)))
#df.drop(delete)
# df.to_csv('../monotonicity_Dataset_For_Train.csv', index=False)
print(len(mem))
print(mem)