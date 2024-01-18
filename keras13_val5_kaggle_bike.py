from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_squared_log_error
import numpy as np
import pandas as pd
import time
###
path = "c://_data//kaggle//bike//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")

train_csv = train_csv.dropna()
test_csv = test_csv.fillna(test_csv.mean())
###
x = train_csv.drop(['count','casual','registered'],axis=1)
y = train_csv['count']
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.75,random_state=1414)

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
model.compile(loss='mae',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=20,
          validation_split=0.15,verbose=1)
loss=model.evaluate(x_test,y_test)
y_submit=model.predict(test_csv)

submission_csv['count']=y_submit
submission_csv.to_csv(path + "sampleSubmission_0108.csv", index=False)
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print("loss:",loss)
print("r2:",r2)
print("음수갯수:",submission_csv[submission_csv['count']<0].count())    ###중요###
def RMSE(a,b):
    # mean_squared_error(y_test,y_predict)
    return np.sqrt(mean_squared_error(a,b))
rmse = RMSE(y_test,y_predict)
print("MSE:", loss)
print("RMSE:",rmse)
