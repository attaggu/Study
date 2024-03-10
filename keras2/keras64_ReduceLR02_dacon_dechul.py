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

path = "C:\\_data\\dacon\\dechul\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0 )
print(train_csv.shape)  
test_csv = pd.read_csv(path + "test.csv", index_col=0 )
print(test_csv.shape) 
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv.shape)  
train_csv=train_csv[train_csv['주택소유상태'] !='ANY']
test_csv.loc[test_csv['대출목적'] == '결혼' , '대출목적'] = '기타'
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder() # 대출기간, 대출목적, 근로기간, 주택소유상태 // 라벨 인코더 : 카테고리형 피처를 숫자형으로 변환
train_csv['주택소유상태'] = le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = le.fit_transform(train_csv['대출목적'])
train_csv['대출기간'] = train_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
train_csv['근로기간'] = le.fit_transform(train_csv['근로기간'])

test_csv['주택소유상태'] = le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = le.fit_transform(test_csv['대출목적'])
test_csv['대출기간'] = test_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
test_csv['근로기간'] = le.fit_transform(test_csv['근로기간'])

train_csv['대출등급'] = le.fit_transform(train_csv['대출등급'])

x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']

y = np.reshape(y, (-1,1)) 

ohe = OneHotEncoder(sparse = False)
ohe.fit(y)
y_ohe = ohe.transform(y)

x_train, x_test, y_train, y_test = train_test_split(
                                                    x,
                                                    y_ohe,             
                                                    train_size=0.88,random_state=2424,
                                                    stratify=y_ohe,
                                                    shuffle=True,
                                                    )
scaler = MinMaxScaler() 
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


model = Sequential()
model.add(Dense(10, input_dim=13, activation='swish'))
model.add(Dense(60, activation='swish')) # 80
model.add(Dense(11, activation='swish'))
model.add(Dense(40, activation='swish'))
model.add(Dense(5, activation='swish'))
model.add(Dense(20, activation='swish'))
model.add(Dense(7, activation='swish'))
model.add(Dense(47, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(7, activation='swish'))
model.add(Dense(35, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(40, activation='swish'))
model.add(Dense(10, activation='swish'))
model.add(Dense(37, activation='swish'))
model.add(Dense(11, activation='swish'))
model.add(Dense(43, activation='swish'))
model.add(Dense(7, activation='softmax'))
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
lr = 0.001


hist=model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr))
hist=model.fit(x_train,y_train, epochs=50, batch_size=500,
          validation_split=0.2,
          callbacks=[es,mcp,rlr],
          )


loss = model.evaluate(x_test,y_test,verbose=0)
y_predict=np.round(model.predict(x_test))
acc = accuracy_score(y_test,y_predict)
print("lr : {0}, loss : {1}".format(lr,loss))
print("lr : {0}, acc : {1}".format(lr, acc))

# lr : 1.0, loss : 0.3532524108886719
# lr : 1.0, acc : 0.0

# lr : 0.1, loss : 0.3530701994895935
# lr : 0.1, acc : 0.0

# lr : 0.01, loss : 0.12356459349393845
# lr : 0.01, acc : 0.7965559016960886

# lr : 0.001, loss : 0.15026378631591797
# lr : 0.001, acc : 0.7627206645898235