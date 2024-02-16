import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,LSTM,Conv1D
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
print(y_train[0])   #5
print(np.unique(y_train,return_counts=True))    #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]


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
filepath= "".join([path,'k31',date,'_',filename]) # ""는 공간을 만든거고 그안에 join으로 합침 , ' _ ' 중간 공간


#3차원 4차원으로 변경
x_train=x_train.reshape(60000,28,28,1)  #data 내용,순서 안바뀌면 reshape 가능

# x_test=x_test.reshape(10000,28,28,1)  #아래와 같다 - 값을 모를때 적용
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
print(x_train.shape[0]) #60000
print(x_train.shape,x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)

y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)
# y_train=to_categorical(y_train)
# y_test=to_categorical(y_test)
ohe=OneHotEncoder(sparse=False)
# y_train=y_train.reshape(-1,1)
# y_test=y_test.reshape(-1,1)
y_train=ohe.fit_transform(y_train)
y_test=ohe.transform(y_test)


# x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,train_size=0.8,random_state=112,
                                            #    stratify=y_train)



#2.model
model = Sequential()
# model.add(LSTM(2,input_shape=(28,28)))
model.add(Conv1D(2,2,input_shape=(28,28)))
model.add(Conv1D(2,2))
model.add(Flatten())
model.add(Dense(17,activation='relu'))
model.add(Dense(14,activation='relu'))
model.add(Dense(8,activation='relu'))
# shape=(행-batch_size,input_dim)
model.add(Dense(10,activation='softmax'))    #숫자를 찾는 분류모델 / (N,27,27,10)
# model.summary()


#3.compile,fit
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es=EarlyStopping(monitor='val_acc',mode='auto',patience=120,restore_best_weights=True)
mcp=ModelCheckpoint(monitor='val_loss',mode='auto',
                    verbose=1,save_best_only=True,
                    filepath=filepath   #경로저장
                    )
model.fit(x_train,y_train,epochs=15,batch_size=88,verbose=1,validation_split=0.2,
          callbacks=[es,mcp])

#4.evaluate,predict
results=model.evaluate(x_test,y_test)
print("loss:",results[0])
print("acc:",results[1])

# loss: 0.1465 - acc: 0.9843    
# loss: 0.1465291827917099     
# acc: 0.9843000173568726  