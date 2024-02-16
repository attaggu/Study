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
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
              [9,8,7,6,5,4,3,2,1,0]])    #[] <= list ("두개이상은")

y=y.T
print(y.shape)
model = Sequential()
model.add(Dense(1, input_dim=3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))


model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=3000, batch_size=2)
loss = model.evaluate(x,y)
result =model.predict([[10,31,211]])
print("loss:", loss)
print("예측값:", result)

    # 예측 : [10, 31, 211]
    # loss: 9.934174143022556e-12
    # 예측값: [[11.000004   2.000001  -1.0000056]]