from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split,KFold,cross_val_score,cross_val_predict
from sklearn.metrics import r2_score,mean_squared_error,mean_squared_log_error,accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
from sklearn.svm import LinearSVR,SVR
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
path = "c://_data//kaggle//bike//"
train_csv=pd.read_csv(path+"train.csv",index_col=0)
test_csv=pd.read_csv(path+"test.csv",index_col=0)
submission_csv=pd.read_csv(path+"sampleSubmission.csv")

train_csv=train_csv.dropna()
test_csv=test_csv.fillna(test_csv.mean())

x=train_csv.drop(['count','casual','registered'],axis=1)
y=train_csv['count']
x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,random_state=121)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

splits = 2
fold = KFold(n_splits=splits,shuffle=True,random_state=950228)
model=SVR()
scores = cross_val_score(model,x,y,cv=fold)
print("acc :",scores , "\ncv-acc :", round(np.mean(scores),4))
y_predict=cross_val_predict(model,x_test,y_test,cv=fold)


