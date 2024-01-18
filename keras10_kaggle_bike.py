from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics import r2_score,mean_squared_error,mean_squared_log_error

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

#1. data

path = "c://_data//kaggle//bike//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)    #1열을 인덱스로잡음
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")
print(submission_csv)

print(train_csv.shape)
print(test_csv.shape)
print(submission_csv.shape)

print(train_csv.columns)
print(test_csv.columns)

print(train_csv.info())
print(test_csv.info())

print(train_csv.describe())
train_csv = train_csv.dropna()



print(train_csv.isna().sum())
print(train_csv.info())
print(train_csv.shape)

test_csv = test_csv.fillna(test_csv.mean())
print(train_csv.info())
print(train_csv.shape)
x = train_csv.drop(['count','casual','registered'],axis=1)
print(x)
y = train_csv['count']
print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.7,
                                                 random_state=1278)
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
model.add(Dense(1)) #마지막 최종 아웃풋레이어에는 'relu'를 잘사용하지 않음(거의 안씀)
#3. compile, fit
model.compile(loss='mae',optimizer='adam')
model.fit(x_train,y_train,epochs=100, batch_size=50)
loss=model.evaluate(x_test,y_test)

y_submit=model.predict(test_csv)
print(y_submit)
print(y_submit.shape)

# y_submit[y_submit<0]=0 #----음수를 전부 0으로 표시
# y_submit=abs(y_submit) ----절대값으로 변경(데이터가 정확하지 않음)

submission_csv['count']=y_submit
print(submission_csv)
submission_csv.to_csv(path + "sampleSubmission_0108.csv", index=False)
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print("loss:",loss)
print("R2_score:",r2)



# print("양의갯수:",submission_csv[submission_csv['count']>0].count())
print("음수갯수:",submission_csv[submission_csv['count']<0].count())    ###중요###
#4. evaluate, predict  6291,202 / 6493,0 / 6289,0




def RMSE(y_test,y_predict):
    # mean_squared_error(y_test,y_predict)
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test,y_predict)
print("MSE:", loss)
print("RMSE:",rmse)

# def RMSLE(y_test,y_predict):
#     return np.sqrt(mean_squared_log_error(y_test,y_predict))    
# rmsle= RMSE(y_test,y_predict)
# print("RMSLE:", rmsle)

