
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

# x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,shuffle=True)

# print(x_train.shape,y_train.shape)  #(397, 10) (397,)
# print(x_test.shape,y_test.shape)    #(45, 10) (45,)


# x_train=x_train.reshape(x_train.shape[0],5,2,1)
# x_test=x_test.reshape(x_test.shape[0],5,2,1)

# y_train=y_train.reshape(-1,1)
# y_test=y_test.reshape(-1,1)
# # ohe=OneHotEncoder(sparse=False)
# # y_train=ohe.fit_transform(y_train)
# # y_test=ohe.fit_transform(y_test)


# print(x_train.shape,y_train.shape)  #(397, 10) (397,)
# print(x_test.shape,y_test.shape)    #(45, 10) (45,)



# scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()

# scaler.fit(x_train)
# x_train=scaler.transform(x_train)
# x_test=scaler.transform(x_test)

# print(np.unique(y_train,return_counts=True))
# print(np.unique(y_test,return_counts=True))


# model=Sequential()
# model.add(Conv2D(13,(2,1),input_shape=(5,2,1),padding='same'))
# model.add(Conv2D(12,(2,1)))
# model.add(Conv2D(12,(2,1)))
# model.add(Flatten)
# model.add(Dense(10,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))

# model.summary()
# es=EarlyStopping(monitor='val_loss',mode='auto',patience=500,verbose=1,
#                  restore_best_weights=True)

# model.compile(loss='bainary_crossentropy', optimizer='adam')
# hist=model.fit(x_train,y_train,epochs=100,batch_size=100,validation_split=0.2,verbose=1,
#                callbacks=[es])

# result=model.evaluate(x_test,y_test)
# y_predict=model.predict(x_test)
# result=model.predict(x)


# # r2=r2_score(y_test,y_predict)
# print("loss:",result[0])
# # print("r2:",r2)

# # # def RMSE(y_test,y_predict):
# # #     return np.sqrt(mean_squared_error(y_test,y_predict))
# # # rmse = RMSE(y_test,y_predict)
# # # print("RMSE:",rmse)
