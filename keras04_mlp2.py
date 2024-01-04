import numpy as np
from keras.models import Sequential
from keras.layers import Dense

x = np.array([[1,2,3,4,5,6,7,8,9,10],
               [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
               [9,8,7,6,5,4,3,2,1,0]
               ])
y = np.array([1,2,3,4,5,6,7,8,9,10])
print(x)
print(x.shape, y.shape)
x =x.T
print(x.shape)

model = Sequential()
model.add(Dense(1, input_dim=3))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=2)

loss = model.evaluate(x, y)
result = model.predict([[5, 1.4, 5]])
print("loss:", loss)
print("[10,1.3,0]:", result)

# loss: 0.005632694344967604
# [10,1.3,0]: [[9.95661]]
