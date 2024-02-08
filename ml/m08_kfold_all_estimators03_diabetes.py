from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split,KFold,cross_val_score,cross_val_predict,StratifiedKFold
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.svm import LinearSVR,SVR,SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import all_estimators
x,y = load_diabetes(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,random_state=1212,train_size=0.8,
                                              )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


splits = 4
fold = KFold(n_splits=splits, shuffle=True, random_state=28)

model = SVR()

scores = cross_val_score(model,x_train,y_train,cv=fold)

print("acc :",scores , "\ncv-acc :", round(np.mean(scores),4))

y_predict=cross_val_predict(model,x_test,y_test,cv=fold)


