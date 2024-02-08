from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import all_estimators
datasets = fetch_california_housing()
x=datasets.data
y=datasets.target
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               train_size=0.8,
                                               random_state=111)



scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

algo = all_estimators(type_filter='regressor')

for name, algori in algo:
    try:
        model = algori()
        model.fit(x_train,y_train)
        acc=model.score(x_test,y_test)
        print(name,'==score:',acc)
    except:
        print(name,'==error')
        continue

'''

# model=LinearSVR()
# model=LinearRegression()
# model=KNeighborsRegressor()
# model=DecisionTreeRegressor()
model=RandomForestRegressor()
# Best RandomForestRegressor


hist=model.fit(x_train,y_train)
loss=model.score(x_test,y_test)
y_predict=model.predict(x_test)
# result=model.predict(x)
r2=r2_score(y_test,y_predict)

def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))
rmse=RMSE(y_test,y_predict)
print("RMSE:",rmse)
print("model.score:",loss)
print("r2",r2)

# RMSE: 0.7386990964209287
# loss: [0.5173534750938416, 0.5456762909889221]
# r2 0.5923106234496522
'''
