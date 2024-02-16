

from keras.models import Sequential,load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
from sklearn.preprocessing import RobustScaler,StandardScaler
from keras.callbacks import EarlyStopping,ModelCheckpoint

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


hist=model=load_model('../_data/_save/MCP/keras26_diabetes_MCP.hdf5')




loss=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
result=model.predict(x)


r2=r2_score(y_test,y_predict)
print("loss:",loss)
print("r2:",r2)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE:",rmse)


