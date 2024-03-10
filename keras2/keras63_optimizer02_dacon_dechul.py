from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
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

from keras.optimizers import Adam
learning_rate = 1.0
model.compile(loss='mae', optimizer=Adam(learning_rate))

hist=model.fit(x_train,y_train, epochs=50, batch_size=500,
          validation_split=0.3)

loss=model.evaluate(x_test, y_test)
# y_submit = model.predict(test_csv)
# submission_csv['count']=y_submit 
# submission_csv.to_csv(path + "submission_0106.csv", index=False)
y_predict=np.round(model.predict(x_test))
acc=accuracy_score(y_test,y_predict)
print("lr : {0}, loss : {1}".format(learning_rate,loss))
print("lr : {0}, loss : {1}".format(learning_rate, acc))

# lr : 1.0, loss : 0.2002175748348236
# lr : 1.0, loss : 0.2992384908272759

# lr : 0.1, loss : 0.20375318825244904
# lr : 0.1, loss : 0.28686396677050885

# lr : 0.01, loss : 0.2002175748348236
# lr : 0.01, loss : 0.2992384908272759

# lr : 0.001, loss : 0.18642140924930573
# lr : 0.001, loss : 0.3475250951886466