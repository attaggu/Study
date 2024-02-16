import numpy as np
from keras.models import Sequential
from keras.layers import Dense

    #1.Data
x = np.array([range(10)])   #range - python에서 제공하는 기본 함수
print(x)    # [[0 1 2 3 4 5 6 7 8 9]]
print(x.shape)  #(1,10)


x = np.array([range(1,10)])   # 1 부터 (10-1)까지
print(x)    # [[1 2 3 4 5 6 7 8 9]]
print(x.shape)  #(1,9)

x = np.array([range(10),range(21,31,),range(201,211)])

print(x)
print(x.shape)
x = x.T
print(x)
print(x.shape)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]])    #[] <= list ("두개이상은")
    # 예측 : [10, 31, 211]
print(y)
print(y.shape)
y=y.T

model = Sequential()
model.add(Dense(1, input_dim=3))
model.add(Dense(2))
model.add(Dense(2))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)
loss = model.evaluate(x, y)
result = model.predict([[10,31,211]])
print("loss:", loss)
print("[10,31,211]:", result)

# loss: 0.11199017614126205
# [10,31,211]: [[10.832105   2.6382966]]

