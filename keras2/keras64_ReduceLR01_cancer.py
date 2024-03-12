from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau



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



# 2. model

model = Sequential()
model.add(Dense(16, input_dim=30))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1,activation='sigmoid'))

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
filepath= "".join([path,'k25',date,'_',filename]) # ""는 공간을 만든거고 그안에 join으로 합침 , ' _ ' 중간 공간



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
hist=model.fit(x_train,y_train, epochs=1000, batch_size=50,
          validation_split=0.2,
          callbacks=[es,mcp,rlr],
          )

loss = model.evaluate(x_test,y_test,verbose=0)
y_predict=model.predict(x_test,verbose=0)
r2 = r2_score(y_predict,y_test)
print("lr : {0}, loss : {1}".format(lr,loss))
print("lr : {0}, acc : {1}".format(lr, r2))

print("====================================")
# print(hist.history['val_loss'])
print("====================================")


#restor best weights
#save best only
#에 대한 고찰

#True,True - 가장 좋은 지점에서 저장 스탑
#True,False - 모든 에포별로 다 저장
#False,True
#False,False - 모두 에포, 안좋은 지점 전부 저장
