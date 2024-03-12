from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

datasets = load_breast_cancer()
x = datasets.data   #.data= x
y = datasets.target #.target= y
import warnings
warnings.filterwarnings('ignore')   #ignore = warnings 안보게할때
print(x.shape)  #(569, 30)
print(y.shape)  #(569,)



x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.9,
                                                 random_state=37,stratify=y)

scaler = MinMaxScaler() 
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


model = Sequential()
model.add(Dense(16, input_dim=30))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1,activation='sigmoid'))
path='../_data/_save/MCP/'  #문자를 저장
filename= '{epoch:04d}-{val_loss:.4f}.hdf5' #0~9999 : 4자리 숫자까지 에포 / 0.9999 소숫점 4자리 숫자까지 발로스
filepath= "".join([path,'k25','_',filename]) # ""는 공간을 만든거고 그안에 join으로 합침 , ' _ ' 중간 공간



es=EarlyStopping(monitor='val_loss',mode='auto',
                 patience=20,verbose=1,restore_best_weights=True,
                 )
mcp=ModelCheckpoint(monitor='val_loss',mode='auto',
                    verbose=1,save_best_only=True,
                    filepath=filepath,   #경로저장
                    period=20,  #20개마다 저장
                    )

rlr = ReduceLROnPlateau(monitor='val_loss', patience=10,mode='auto',verbose=1,
                        factor=0.5, #중간에 러닝레이트를 반으로 줄인다 / 디폴트 0.001 - 추천 x 
                        )

from keras.optimizers import Adam
lr = 0.01


hist=model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr))
hist=model.fit(x_train,y_train, epochs=10, batch_size=50,
          validation_split=0.2,
         
          )


loss = model.evaluate(x_test,y_test,verbose=0)
y_predict=model.predict(x_test,verbose=0)
r2 = r2_score(y_predict,y_test)
print("lr : {0}, loss : {1}".format(lr,loss))
print("lr : {0}, loss : {1}".format(lr, r2))


# lr : 0.01, loss : 0.027235252782702446
# lr : 0.01, loss : 0.9797003665209393