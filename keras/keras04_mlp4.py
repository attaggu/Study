import numpy as np
from keras.models import Sequential
from keras.layers import Dense

    #1.Data
x = np.array([range(10)])   #range - python에서 제공하는 기본 함수
print(x)    
print(x.shape)  

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
              [9,8,7,6,5,4,3,2,1,0]
              ])
print(y)
print(y.shape)
x=x.T
y=y.T
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(3))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=300, batch_size=1)
loss = model.evaluate(x,y)
result = model.predict([[10]])
print("loss:", loss)
print("[10]:", result)

# loss: 1.93110221147208e-08
# ???: [[10.999905   1.9999952 -0.9997684]]