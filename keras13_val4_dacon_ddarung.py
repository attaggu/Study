from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
import time


path = "c:\\_data\\dacon\\ddarung\\"
train_csv = pd.read_csv(path + "train.csv",index_col=0)
test_csv = pd.read_csv(path + "test.csv",index_col=0)
submission_csv = pd.read_csv(path + "submission.csv")
train_csv = train_csv.dropna() 
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'],axis=1)
y = train_csv['count'] 
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,random_state=563)

model=Sequential()
model.add(Dense(9, input_dim=9))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train, epochs=1000, batch_size=100,
          validation_split=0.3,verbose=1)
loss=model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
submission_csv['count']=y_submit 
submission_csv.to_csv(path + "submission_0105.csv", index=False)
y_predict=model.predict(x_test)

r2 = r2_score(y_test,y_predict)
print("loss:", loss)
print("R2_score:",r2)
# 7/7 [==============================] - 0s 3ms/step - loss: 39.7985 - val_loss: 41.7087
# 11/11 [==============================] - 0s 511us/step - loss: 38.8784
# 23/23 [==============================] - 0s 468us/step
# 11/11 [==============================] - 0s 539us/step
# loss: 38.878421783447266
# R2_score: 0.5437318238772592
# PS C:\Study\keras> 