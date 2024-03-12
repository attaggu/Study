from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import pandas as pd

path = "c://_data//kaggle//bike//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)    #1열을 인덱스로잡음
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")
train_csv = train_csv.dropna()
test_csv = test_csv.fillna(test_csv.mean())
x = train_csv.drop(['count','casual','registered'],axis=1)
y = train_csv['count']


x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.9,
                                                 random_state=37)



# 2. model

model=Sequential()
model.add(Dense(64, input_dim=8,activation='relu'))   #model.add(Dense(64, input_dim=8, activation='relu')) - activation활성화함수
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))
path='../_data/_save/MCP/'  #문자를 저장
filename= '{epoch:04d}-{val_loss:.4f}.hdf5' #0~9999 : 4자리 숫자까지 에포 / 0.9999 소숫점 4자리 숫자까지 발로스
filepath= "".join([path,'k25','_',filename])

es=EarlyStopping(monitor='val_loss',mode='auto',
                 patience=20,verbose=1,restore_best_weights=True,
                 )
mcp=ModelCheckpoint(
    monitor='val_loss',mode='auto',
                    verbose=1,save_best_only=True,
                    filepath= filepath,
                    period=20,  #20개마다 저장
                    )

rlr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1,
                        factor=0.5, #중간에 러닝레이트를 반으로 줄인다 / 디폴트 0.001 - 추천 x 
                        )

from keras.optimizers import Adam
lr = 0.001


model.compile(loss='mae',optimizer=Adam(lr))
model.fit(x_train,y_train,epochs=50, batch_size=50,validation_split=0.2,
          callbacks=[es,mcp,rlr])
loss=model.evaluate(x_test,y_test)

y_submit=model.predict(test_csv)
print(y_submit)
print(y_submit.shape)

# y_submit[y_submit<0]=0 #----음수를 전
submission_csv['count']=y_submit
print(submission_csv)
submission_csv.to_csv(path + "sampleSubmission_0108.csv", index=False)
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print("lr : {0}, loss : {1}".format(lr,loss))
print("lr : {0}, loss : {1}".format(lr, r2))


print("양의갯수:",submission_csv[submission_csv['count']>0].count())

# lr : 1.0, loss : 139.1914520263672
# lr : 1.0, loss : -0.05577315361767088

# lr : 0.1, loss : 139.15194702148438
# lr : 0.1, loss : -0.0661927444390944

# lr : 0.01, loss : 151.10736083984375
# lr : 0.01, loss : -0.4520174062429252

# lr : 0.001, loss : 184.2278289794922
# lr : 0.001, loss : -1.0195238849864294