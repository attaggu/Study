import numpy as np
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D,Input
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,StandardScaler
from keras.callbacks import EarlyStopping,ModelCheckpoint


#1.data
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape,y_train.shape)  #(60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape)    #(10000, 28, 28) (10000,)
# print(x_train)
# print(x_train[0])
print(np.unique(y_train,return_counts=True)) 

#scaler 1-1
# x_train = x_train/255.
# x_test = x_test/255.

# #scaler 1-2
x_train = (x_train-127.5)/127.5
x_test = (x_test-127.5)/127.5

# #scaler 2-1
# scaler=MinMaxScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.fit_transform(x_test)

# #scaler 2-2
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.fit_transform(x_test)


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

print(np.max(x_train),np.min(x_test))

#이미지에서 MinMax 가장 많이 사용

input1=Input(shape=(784,))
dense1=Dense(100)(input1)
drop1=Dropout(0.5)(dense1)
dense2=Dense(200)(drop1)
drop2=Dropout(0.5)(dense2)
dense3=Dense(130)(drop2)
dense4=Dense(60)(dense3)
output1=Dense(10,activation='softmax')(dense4)
model=Model(inputs=input1,outputs=output1)


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=50,batch_size=800,validation_split=0.2)
result=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
print("loss:",result[0])
print("acc:",result[1])

# loss: 0.3308744728565216
# acc: 0.9128000140190125
#적용
# loss: 0.2835792601108551
# acc: 0.9192000031471252  