from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
datasets = load_boston()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.75,
                                                 random_state=12345)
model = Sequential()
model.add(Dense(16, input_dim=13))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1000,batch_size=20,
          validation_split=0.3,verbose=1)
loss = model.evaluate(x_test,y_test)

y_predict=model.predict(x_test)
result=model.predict(x)
r2=r2_score(y_test,y_predict)
print("loss:",loss)
print("r2:",r2)

# 14/14 [==============================] - 0s 2ms/step - loss: 24.5539 - val_loss: 30.9001
# 4/4 [==============================] - 0s 680us/step - loss: 24.7591
# 4/4 [==============================] - 0s 667us/step
# 16/16 [==============================] - 0s 467us/step
# loss: 24.75914764404297
# r2: 0.6149876572231111