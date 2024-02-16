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




model = Sequential()
model.add(Dense(10,input_dim=8))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mae',optimizer='adam',metrics=['mse'])
# es=EarlyStopping(monitor='val_loss',mode='min',
                #  patience=20,verbose=1,
                #  restore_best_weights=True)

hist=model.fit(x_train,y_train,epochs=400,batch_size=100,
            #    callbacks=[es],
               validation_split=0.2)

loss=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
result=model.predict(x)
r2=r2_score(y_test,y_predict)

def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))
rmse=RMSE(y_test,y_predict)
print("RMSE:",rmse)
print("loss:",loss)
print("r2",r2)

# RMSE: 0.7386990964209287
# loss: [0.5173534750938416, 0.5456762909889221]
# r2 0.5923106234496522