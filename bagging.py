import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

f1 = np.array(pd.read_csv('./compare/lgb_meandata_results.csv'))
print(f1.shape)
f2 = np.array(pd.read_csv('./compare/lgb_fulldata_results.csv'))
print(f2.shape)
f3 = np.array(pd.read_csv('./compare/nn_meandata_results.csv'))
print(f3.shape)
f4 = np.array(pd.read_csv('./compare/nn_fulldata_results.csv'))
print(f4.shape)
f5 = np.array(pd.read_csv('./compare/rf_meandata_results.csv'))
print(f5.shape)
f6 = np.array(pd.read_csv('./compare/rf_fulldata_results.csv'))
print(f6.shape)
label = np.array(pd.read_csv('./compare/bagging_label.csv'))
print(label.shape)
X = np.hstack((f1-f6, f2-f6, f3-f6, f4-f6, f5-f6))
y = label-f6

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression(fit_intercept=False)
reg = model.fit(X_train, y_train)
args = reg.coef_
W = args.tolist()
W.append(1-args.sum())
print('[lgb_meandata:lgb_fulldata:nn_meandata:nn_fulldata:rf_meandata:rf_fulldata]:',W)

print('Mean squared error: %.3f' % mean_squared_error(y_test,model.predict(X_test)))
print('score: %.3f' % model.score(X_test,y_test))
# plt.scatter(X_test , y_test ,color ='green')
# plt.plot(X_test ,model.predict(X_test) ,color='red',linewidth =3)
# plt.show()