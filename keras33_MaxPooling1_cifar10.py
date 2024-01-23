from keras.datasets import cifar10
import numpy as np
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping,ModelCheckpoint

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

print(x_train.shape,y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape)    #(10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train,return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],

ohe = OneHotEncoder(sparse=False)
y_train=ohe.fit_transform(y_train)
y_test=ohe.fit_transform(y_test)
# ohe=OneHotEncoder(sparse=False)
# y_train=y_train.reshape(-1,1)
# y_test=y_test.reshape(-1,1)
# y_train=ohe.fit_transform(y_train)
# y_test=ohe.fit_transform(y_test)
x_train=x_train/255
x_test=x_test/255
import datetime
date=datetime.datetime.now()
print(date) #2024-01-17 10:54:36.094603 - 
#월,일,시간,분 정도만 추출
print(type(date))   #<class 'datetime.datetime'>
date=date.strftime("%m%d-%H%M")
#%m 하면 month를 땡겨옴, %d 하면 day를 / 시간,분은 대문자
print(date) #0117_1058
print(type(date))   #<class 'str'> 문자열로 변경됨

path='../_data/_save/MCP/'  #문자를 저장
filename= '{epoch:04d}-{val_loss:.4f}.hdf5' #0~9999 : 4자리 숫자까지 에포 / 0.9999 소숫점 4자리 숫자까지 발로스
filepath= "".join([path,'k31_5',date,'_',filename])

model=Sequential()
model.add(Conv2D(88,(3,3),input_shape=(32,32,3),
                 strides=2,
                #  padding='same',
                padding='valid'
                 ))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(24,(2,2),activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D())
model.add(Conv2D(70,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(37,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(55,activation='relu'))
model.add(Dense(10,activation='softmax'))





model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics=['acc'])
es=EarlyStopping(monitor='val_loss',mode='auto',patience=400,restore_best_weights=True)
mcp=ModelCheckpoint(monitor='val_loss',mode='auto',
                    verbose=1,save_best_only=True,
                    filepath=filepath)
model.fit(x_train,y_train,epochs=20,batch_size=1000,verbose=1,validation_split=0.15,
          callbacks=[es,mcp])
result=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)

print("loss:",result[0])
print("acc:",result[1])