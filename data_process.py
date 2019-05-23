import pandas as pd
import numpy as np

###add ave_Delivery_time feature
"""
df1 = pd.read_csv('../Dataset_For_Train.csv')
# df1 = pd.read_csv('../TestB.csv')
c = list(df1['Delivery_time'])
N = len(df1)
k = 0
for i in range(N):
    if i%(N//10)==0:
        print('-----finish %d0%%-----'%(k))
        k += 1
    l = list(map(int,c[i].strip('"').split(',')))
    c[i] = sum(l)//7

df1['ave_Delivery_time'] = c
df1.to_csv('../new.csv', index=False)
# df1.to_csv('../TestB.csv',index=False)


"""

###record ages, genders, areas respectly
# df = pd.read_csv('../TestB_Sample_Data.csv')
df = pd.read_csv('../new.csv')
N = len(df)
print(N)
Age = [np.nan for _ in range(N)]
Gender = [np.nan for _ in range(N)]
Area = [np.nan for _ in range(N)]
l = list(df['Chose_People'])
k = 0
for i in range(N):
    if i % (N// 10) == 0:
        print('-----finish %d0%%-----'%(k))
        k += 1
    p1 = 0
    p2 = 0
    s = l[i]
    if 'age' in s:
        while s[p1:p1+3] != 'age':
            p1 += 1
        if '|' in s:
            p2 = s.index('|')
        else:
            p2 = len(s)
        Age[i] = list(map(int,s[p1+4:p2].strip().split(',')))
    if 'gender' in s[p2:]:
        p1 = p2
        while s[p1:p1+6] != 'gender':
            p1 += 1
        if '|' in s[p1:]:
            p2 = p1 + s[p1:].index('|')
        else:
            p2 = len(s)
        Gender[i] = list(map(int,s[p1+7:p2].strip().split(',')))
    if 'area' in s[p2:]:
        p1 = p2
        while s[p1:p1 + 4] != 'area':
            p1 += 1
        if '|' in s[p1:]:
            p2 = p1 + s[p1:].index('|')
        else:
            p2 = len(s)
        # df['Area'][i] = list(map(int, s[p1+5:p2].split(',')))
        temp = s[p1+7:p2].strip().split(',')
        #ValueError: invalid literal for int() with base 10: ''说明存在，，
        while '' in temp:
            temp.remove('')
        Area[i] = list(map(int,temp))
df['age'] = Age
df['gender'] = Gender
df['area'] = Area
df.to_csv('../new.csv', index=False)
#df.to_csv('../TestB_Sample_Data.csv', index=False)

