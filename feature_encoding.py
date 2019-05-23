from collections import Counter
import pandas as pd
import numpy as np

def encoding(feature,chosen_list):
    s = ''
    for n in chosen_list:
        if str(n) in feature:
            s += '1'
        else:
            s += '0'
    return int(s, 2)

df = pd.read_csv('../new.csv')
df_test = pd.read_csv('../TestB_Sample_Data.csv')
Age = df['age']
Gender = df['gender']
Area = df['area']
###
Age_test = df_test['age']
Gender_test = df_test['gender']
Area_test = df_test['area']
###
l_age = []
l_area = []
for i in range(len(df)):
    if Age[i] is not np.nan:
        l_age.extend(list(map(int,Age[i].strip('[]').split(','))))
    if Area[i] is not np.nan:
        l_area.extend(list(map(int,Area[i].strip('[]').split(','))))
counter_age = Counter(l_age)
counter_area = Counter(l_area)
candidates_age = list(counter_age.keys())
candidates_area = list(counter_area.keys())
candidates_age.sort(key = lambda x:-counter_age[x])
candidates_area.sort(key = lambda x:-counter_area[x])
chosen_Ages = candidates_age[:20]
chosen_Areas = candidates_area[:30]
chosen_Genders = [3,2,1]
print('the highest frequency of 20 ages:', chosen_Ages)
print('the highest frequency of 30 areas:', chosen_Areas)
All_Age = int('11111111111111111111',2)
All_Area = int('111111111111111111111111111111',2)
All_Gender = int('111',2)

N = len(df)
Age_coding = [All_Age for _ in range(N)]
Gender_coding = [All_Gender for _ in range(N)]
Area_coding = [All_Area for _ in range(N)]

l = list(df['Chose_People'])
k = 0
for i in range(N):
    if i % (N // 10) == 0:
        print('-----finish %d0%%-----'%(k))
        k += 1
    if l[i] == 'all':
        continue
    else:
        if Age[i] is not np.nan:
            Age_coding[i] = encoding(Age[i],chosen_Ages)
        if Gender[i] is not np.nan:
            Gender_coding[i] = encoding(Gender[i],chosen_Genders)
        if Area[i] is not np.nan:
            Area_coding[i] = encoding(Area[i],chosen_Areas)
df['age_feature'] = Age_coding
df['gender_feature'] = Gender_coding
df['area_feature'] = Area_coding
print(len(df))
df.to_csv('../new.csv', index=False)

###
N = len(df_test)
Age_coding = [All_Age for _ in range(N)]
Gender_coding = [All_Gender for _ in range(N)]
Area_coding = [All_Area for _ in range(N)]
l = list(df_test['Chose_People'])
k = 0
for i in range(N):
    if i % (N // 10) == 0:
        print('-----finish %d0%%-----'%(k))
        k += 1
    if l[i] == 'all':
        continue
    else:
        if Age_test[i] is not np.nan:
            Age_coding[i] = encoding(Age_test[i],chosen_Ages)
        if Gender_test[i] is not np.nan:
            Gender_coding[i] = encoding(Gender_test[i],chosen_Genders)
        if Area_test[i] is not np.nan:
            Area_coding[i] = encoding(Area_test[i],chosen_Areas)
df_test['age_feature'] = Age_coding
df_test['gender_feature'] = Gender_coding
df_test['area_feature'] = Area_coding
# df.to_csv('../new_Dataset_For_Train.csv', index=False)
df_test.to_csv('../TestB.csv', index=False)
