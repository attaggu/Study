from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import all_estimators
path = "c:\\_data\\dacon\\ddarung\\"
train_csv=pd.read_csv(path + "train.csv",index_col=0)
test_csv=pd.read_csv(path + "test.csv",index_col=0)
submission_csv =pd.read_csv(path +"submission.csv")
train_csv=train_csv.dropna()
test_csv=test_csv.fillna(test_csv.mean())

x=train_csv.drop(['count'],axis=1)
y=train_csv['count']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=121)


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

submission_csv['count']=model.predict(test_csv)

submission_csv.to_csv(path + "submission_0116.csv",index=False)
y_predict=model.predict(x_test)

r2=r2_score(y_test,y_predict)
print("model.score:",loss)
print("r2:",r2)
def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))
rmse=RMSE(y_test,y_predict)
print("RMSE:",rmse)

'''
