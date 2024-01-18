from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. data

x = np.array([1,2,3])
y = np.array([1,2,3])

#2. model

model = Sequential()
model.add(Dense(3, input_dim=1))123
model.add(Dense(1))
model.add(Dense(2))
model.add(Dense(11))
model.add(Dense(22))
model.add(Dense(33))
model.add(Dense(44))
model.add(Dense(55))
model.add(Dense(66))
model.add(Dense(77))
model.add(Dense(88))
model.add(Dense(99))
model.add(Dense(88))
model.add(Dense(77))
model.add(Dense(66))
model.add(Dense(55))
model.add(Dense(44))
model.add(Dense(33))
model.add(Dense(22))
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))
# model = Sequential()
# model.add(Dense(3, input_dim=1))
# model.add(Dense(6, input_dim=3))
# model.add(Dense(4, input_dim=6))
# model.add(Dense(2, input_dim=4))
# model.add(Dense(1, input_dim=2))
# same
# model = Sequential()
# model.add(Dense(3, input_dim=1))
# model.add(Dense(6))
# model.add(Dense(4))
# model.add(Dense(2))
# model.add(Dense(1))

#3 compile, fit

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

#evaluate, 

loss = model.evaluate(x, y)
print("loss : ", loss)
result = model.predict([4])
print("4의 예측값 : ", result)


# loss :  1.4958071005821694e-05
# 1/1 [==============================] - 0s 115ms/step
# 4의 예측값 :  [[4.0000772]]