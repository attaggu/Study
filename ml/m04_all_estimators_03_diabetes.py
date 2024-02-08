from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import all_estimators
datasets=load_diabetes()
x=datasets.data
y=datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=123)


scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

Algor = all_estimators(type_filter='regressor')
for name, algorithms in Algor:
    try:
        model = algorithms()
        model.fit(x_train,y_train)
        
        acc=model.score(x_test,y_test)
        print(name, '==score:', acc)
    except:
        print(name, '==error')
        continue

'''
# model=LinearSVR()
model=LinearRegression()
# model=KNeighborsRegressor()
# model=DecisionTreeRegressor()
model=RandomForestRegressor()
# Best LinearRegression RandomForestRegressor


hist=model.fit(x_train,y_train)
loss=model.score(x_test,y_test)
y_predict=model.predict(x_test)
result=model.predict(x)
r2=r2_score(y_test,y_predict)
print("model.score:",loss)
print("r2:",r2)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE:",rmse)


# plt.rcParams['font.family']='Malgun Gothic'
# plt.rcParams['axes.unicode_minus']=False

# plt.figure(figsize=(10,10))
# plt.legend(loc='upper right')
# plt.title('당뇨병')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

# loss: 2306.89501953125
# r2: 0.6540809404569466
# RMSE: 48.0301477131255

'''
