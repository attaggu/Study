import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPool2D,LSTM,Conv1D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

np_path='c:/_data/_save_npy//'

x=np.load(np_path+'keras39_9_x_train.npy')
y=np.load(np_path+'keras39_9_y_train.npy')

print(x.shape,y.shape)  #(2520, 150, 150, 3) (2520, 3)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=123,
                                                 train_size=0.8,stratify=y) 

x_train=x_train.reshape(-1,150,150*3)
x_test=x_test.reshape(-1,150,150*3)

model=Sequential()
# model.add(Conv2D(11,(2,2),input_shape=(150,150,3)))
# model.add(LSTM(10,input_shape=(150,150*3)))
model.add(Conv1D(5,2,input_shape=(150,150*3)))
model.add(Conv1D(2,2))
model.add(Flatten())
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(3,activation='softmax'))

es=EarlyStopping(monitor='val_loss',mode='auto',patience=100,
                 restore_best_weights=True)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
hist=model.fit(x_train,y_train,epochs=30,batch_size=32,verbose=1,
               validation_split=0.15,callbacks=[es])
result=model.evaluate(x_test,y_test)

y_predict=model.predict(x_test)
argy_predict=np.argmax(y_predict,axis=1)    #제출목적이 아니면 사용 안해도 무관
# argy_predict=np.around(y_predict) 제출목적이 아니면 사용 안해도 무관

print("loss:",result[0])
print("acc:",result[1])
print(argy_predict)
# loss: 0.044129628688097
# acc: 0.9900793433189392