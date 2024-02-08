from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold,cross_val_predict
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.svm import LinearSVR,SVR
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
x,y=fetch_california_housing(return_X_y=True)
# datasets= fetch_california_housing()
# x=np.array(datasets.data)
# y=np.array(datasets.target)

x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,random_state=121)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

splits=5
# fold = KFold(n_splits=splits,shuffle=True,random_state=119)
fold = KFold(n_splits=splits,shuffle=True,random_state=119)

model = RandomForestRegressor()

scores=cross_val_score(model,x_train,y_train,cv=fold)

print("acc :",scores , "\ncv-acc :", round(np.mean(scores),4))
y_predict=cross_val_predict(model,x_test,y_test,cv=fold)
