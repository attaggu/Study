from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
import time

datasets=load_diabetes()
x=datasets.data
y=datasets.target
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,random_state=166)

model = Sequential()
model.add(Dense(10,input_dim=10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
start_time=time.time()
model.fit(x_train,y_train,epochs=500,batch_size=100,
          validation_split=0.3,verbose=1)
end_time=time.time()
loss=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
result=model.predict(x)
r2=r2_score(y_test,y_predict)
print("loss:",loss)
print("r2:",r2)
print("time:",round(end_time-start_time,2),"s")

# 3/3 [==============================] - 0s 8ms/step - loss: 2733.1826 - val_loss: 3174.5874
# 4/4 [==============================] - 0s 747us/step - loss: 3101.6541
# 4/4 [==============================] - 0s 609us/step
# 14/14 [==============================] - 0s 430us/step
# loss: 3101.654052734375
# r2: 0.436635328223426
# time: 8.21 s