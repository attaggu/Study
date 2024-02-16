import numpy as np
from keras.models import Sequential
from keras.layers import Dense
x = np.array([[1,2,3,4,5],
             [6,7,8,9,10],
             [11,12,13,14,15]])
y = np.array([100,200,300,400,500])
print(x.shape)
print(y.shape)

x = x.T

model = Sequential()
model.add(Dense(1, input_dim=3))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=2)
loss = model.evaluate(x, y)
result = model.predict([[2,7,12]])
print("loss:", loss)
print("??:", result)
