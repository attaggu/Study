import numpy as np
from keras.models import Sequential
from keras.layers import Dense

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(x,y, epochs=1000, batch_size=2)
loss = model.evaluate(x,y)
result = model.predict([11000,7])
print("loss:", loss)
print("??:", result)
# loss: 5.196712749011567e-08
# ??: [[1.1000151e+04]
#  [7.0002451e+00]]
# 2/2 [==============================] - 0s 858us/step - loss: 0.2007
# 1/1 [==============================] - 0s 69ms/step - loss: 0.1989
# 1/1 [==============================] - 0s 56ms/step
# loss: 0.19890935719013214
# ??: [[1.0887138e+04]
#  [6.9942346e+00]]