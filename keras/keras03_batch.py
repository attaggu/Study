# from tensorflow.keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import keras
print("tf version: ", tf.__version__)
print("keras version: ", keras.__version__)

# 1. data
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# 2. model
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(11))
model.add(Dense(22))
model.add(Dense(33))
model.add(Dense(44))
model.add(Dense(55))
model.add(Dense(66))
model.add(Dense(77))
model.add(Dense(88))
model.add(Dense(99))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
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
model.add(Dense(1))

# 3. compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=2)
# batch_size = ?? 데이터값을 한개씩 따로한다 = 너무 많은 데이터가 있을때 잘라서 작업을 돌려서 터짐을 방지(얼만큼 한번에 돌리겠냐?) / 평균 batch size 32개

# 4. evaluate, result
loss = model.evaluate(x, y)
result = model.predict([7])
print("loss:", loss)
print("7의 예측값:", result)

#epochs=100 batch_size=2
# loss: 0.32758355140686035
# 7의 예측값: [[6.6644554]]
