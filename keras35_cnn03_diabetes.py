
from keras.models import Sequential,load_model,Model
from keras.layers import Dense,Dropout,Input,Conv2D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,OneHotEncoder
from sklearn.preprocessing import RobustScaler,StandardScaler
from keras.callbacks import EarlyStopping,ModelCheckpoint
import datetime
datasets=load_diabetes()
x=datasets.data
y=datasets.target
print(np.unique(y,return_counts=True))

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,shuffle=True)

print(x_train.shape,y_train.shape)  #(397, 10) (397,)
print(x_test.shape,y_test.shape)    #(45, 10) (45,)




# y_train=y_train.reshape(-1,1)
# y_test=y_test.reshape(-1,1)
# ohe=OneHotEncoder(sparse=False)
# y_train=ohe.fit_transform(y_train)
# y_test=ohe.fit_transform(y_test)


print(x_train.shape,y_train.shape)  #(397, 10) (397,)
print(x_test.shape,y_test.shape)    #(45, 10) (45,)



# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

print(np.unique(y_train,return_counts=True))
print(np.unique(y_test,return_counts=True))
x_train=x_train.reshape(-1,5,2,1)
x_test=x_test.reshape(-1,5,2,1)

model=Sequential()
model.add(Conv2D(13,(2,2),input_shape=(5,2,1)))
model.add(Flatten())
model.add(Dense(10,activation='swish'))
model.add(Dense(10,activation='swish'))
model.add(Dense(10,activation='swish'))
model.add(Dense(10,activation='swish'))
model.add(Dense(1))

model.summary()
es=EarlyStopping(monitor='val_loss',mode='auto',patience=500,verbose=1,
                 restore_best_weights=True)

model.compile(loss='mse', optimizer='adam')
hist=model.fit(x_train,y_train,epochs=1000,batch_size=1000,validation_split=0.2,verbose=1,
               callbacks=[es])

result=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)

r2=r2_score(y_test,y_predict)
print("loss:",result)
print("r2:",r2)

# # def RMSE(y_test,y_predict):
# #     return np.sqrt(mean_squared_error(y_test,y_predict))
# # rmse = RMSE(y_test,y_predict)
# # print("RMSE:",rmse)
