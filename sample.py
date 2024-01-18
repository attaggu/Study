import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([1,2,3,4,5,6,7,8,9])

model = Sequential()
model.add(Dense(9, input_dim=1))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

loss = model.evaluate(x, y)
print("loss : ", loss)
result = model.predict([10])
print("10의 예측값 : ", result)

