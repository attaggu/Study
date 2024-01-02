import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1.data
x = np.array([[1,2,3,4,5,6,7,8,9,10], 
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]
             )
y = np.array([1,2,3,4,5,6,7,8,9,10])
print(x.shape)  #(2, 10)
print(y.shape)  #(10,) - 스칼라 10개짜리 벡터가 1개
x = x.T #   전치 => 행과 열을 변경 => x = x.transpose()
#[[1,1],[2,1.1],[3,1.2], ...[10,1.3]]
print(x.shape) #(10, 2)

#[[[[1,2,3],[4,5,6]]]] = 1,1,2,3

# 2.model

model = Sequential()
model.add(Dense(1, input_dim=2)) #열, 컬럼, 속성, 특성,차원 = 2로 같다.
 #  &&&&(행무시, 열우선)
model.add(Dense(1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))
 
 # 3.compile,fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=5)

 # 4.evauate, result
loss = model.evaluate(x, y)
result = model.predict([[10, 1.3]]) #   shape - 행무시, 열우선
print("loss:", loss)
print("[10, 1.3]:", result)

# loss: 0.002725246362388134
# [10, 1.3]: [[10.076436]]