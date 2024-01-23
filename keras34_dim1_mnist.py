import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping,ModelCheckpoint


#1.data
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape,y_train.shape)  #(60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape)    #(10000, 28, 28) (10000,)
# print(x_train)
# print(x_train[0])
print(np.unique(y_train,return_counts=True)) 
y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1) 
ohe = OneHotEncoder(sparse=False)   #spares=True 가 디폴트
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
# fit + transform 대신 쓴다 
# ==== ohe.fit(y)
#3차원 4차원으로 변경
# x_train=x_train.reshape(60000,28,28,1) 2차원으로 변경

x_train=x_train.reshape(60000,28*28)
x_test=x_test.reshape(10000,28*28)
# (60000, 784) (10000, 784)
print(x_train.shape,x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)
'''
model=Sequential()
model.add(Dense(100,input_shape=(784,)))
model.add(Dropout(0.5))
model.add(Dense(200))
model.add(Dropout(0.5))
model.add(Dense(130))
model.add(Dense(60))
model.add(Dense(10,activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=50,batch_size=800,validation_split=0.2)
result=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
print("loss:",result[0])
print("acc:",result[1])
'''