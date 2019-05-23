import pandas as pd
import numpy as np
# df = pd.read_csv('../TestB.csv')
# df1 = pd.read_csv('../B.csv')
# l = df1['bid']
# df['bid'] = l
# df.to_csv('../TestB.csv', index=False)

######
w = 0.0224
# df = pd.read_csv('../Dataset_For_Train_mean.csv')
df = pd.read_csv('./data/Dataset_For_Train_mean_filtered.csv')
bids = np.array(list(df['ad_bid']))

# Dvalue = w*bids
# np.savetxt('D_vakues.csv', Dvalue, delimiter = ',')

clicks = np.array(list(df['num_click']))
labels = (clicks - w*bids).tolist()
labels_df = pd.DataFrame(labels, columns=['label'])
labels_df.to_csv('../mean_labels_w=0.0224.csv')
######

###save tarin_data as array pattern
# df = pd.read_csv('../new.csv')
# df1 = pd.read_csv('../labels_w=0.0224.csv')
data = []
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
    data.append(l)
data = np.array(data, dtype='int64')
labels = labels_df['label']
target = []
for j in range(len(labels_df)):
    target.append(labels[j])
target = np.array(target)
np.savetxt('mean_train_data.csv', data, delimiter = ',')
np.savetxt('mean_train_label.csv', target, delimiter = ',')