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

path = "c:\\_data\\dacon\\ddarung\\"
train_csv = pd.read_csv(path + "train.csv",index_col=0)
test_csv = pd.read_csv(path + "test.csv",index_col=0)
submission_csv = pd.read_csv(path + "submission.csv")
train_csv = train_csv.dropna() 
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'],axis=1)
y = train_csv['count'] 
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,random_state=2086)


# 2. model

model=Sequential()
model.add(Dense(7, input_dim=9,activation='relu'))
model.add(Dense(7))
# model.add(Dense(11,activation='relu'))
# model.add(Dense(11,activation='relu'))
model.add(Dense(11))
# model.add(Dense(11,activation='relu'))
# model.add(Dense(11,activation='relu'))
model.add(Dense(11))
# model.add(Dense(11,activation='relu'))
# model.add(Dense(11,activation='relu'))
# model.add(Dense(7,activation='relu'))
# model.add(Dense(7,activation='relu'))
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
lr = 1.0


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

# lr : 1.0, loss : 62.32895278930664
# lr : 1.0, loss : -0.0004965554141704853

# lr : 0.1, loss : 38.36003112792969
# lr : 0.1, loss : 0.5663297258409288

# lr : 0.01, loss : 38.962215423583984
# lr : 0.01, loss : 0.5498577994978003

# lr : 0.001, loss : 54.27073669433594
# lr : 0.001, loss : 0.13411503016589066