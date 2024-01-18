from keras.models import Sequential,load_model,Model
from keras.layers import Dense,Dropout,Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
import datetime
import time
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




# model = Sequential()
# model.add(Dense(10,input_shape=(8,)))
# model.add(Dense(10))
# model.add(Dropout(0.2))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dropout(0.2))
# model.add(Dense(10))
# model.add(Dense(1))

input1=Input(shape=(8,))
dense1=Dense(10)(input1)
dense2=Dense(10)(dense1)
drop1=Dropout(0.2)(dense2)
dense3=Dense(10)(drop1)
dense4=Dense(10)(dense3)
drop2=Dropout(0.2)(dense4)
dense5=Dense(10)(drop2)
output1=Dense(1)(dense5)
model=Model(inputs=input1,outputs=output1)


date=datetime.datetime.now()
date=date.strftime("%m%d-%H%M")
path='../_data/_save/MCP/'
filename='{epoch:04d}-{val_loss:4f}.hdf5'
filepath="".join([path,'k26_02_california_',date,'_',filename])

hist=model.compile(loss='mae',optimizer='adam',metrics=['mse'])
es=EarlyStopping(monitor='val_loss',mode='min',
                 patience=20,verbose=1,
                 restore_best_weights=True)
mcp=ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
                    save_best_only=True,
                    filepath=filepath
                    )
start_time=time.time()
hist=model.fit(x_train,y_train,epochs=400,batch_size=100,
               callbacks=[es,mcp],
               validation_split=0.2)
end_time=time.time()

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
print("time:",round(end_time-start_time,2),"s")
# RMSE: 0.7386990964209287
# loss: [0.5173534750938416, 0.5456762909889221]
# r2 0.5923106234496522