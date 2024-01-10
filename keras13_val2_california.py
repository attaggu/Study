from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import time

datasets = fetch_california_housing()
x=datasets.data
y=datasets.target
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,random_state=950228)

model = Sequential()
model.add(Dense(16, input_dim=8))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
start_time=time.time()
model.fit(x_train,y_train,epochs=1000,batch_size=100,
          validation_split=0.3,verbose=1)
end_time=time.time()
loss=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
result=model.predict(x)
r2=r2_score(y_test,y_predict)
print("loss:",loss)
print("r2:",r2)
print("time",round(end_time-start_time,2),"s")


# 109/109 [==============================] - 0s 842us/step - loss: 0.6235 - val_loss: 0.5816
# 162/162 [==============================] - 0s 417us/step - loss: 0.6220
# 162/162 [==============================] - 0s 395us/step
# 645/645 [==============================] - 0s 410us/step
# loss: 0.6219624280929565
# r2: 0.5452593418712428
# time 89.84 s