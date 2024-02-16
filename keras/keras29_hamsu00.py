import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,Input,Dropout

# 1.data
x = np.array([[1,2,3,4,5,6,7,8,9,10], 
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]
             )
y = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape)  #(2, 10)
# print(y.shape)  #(10,) - 스칼라 10개짜리 벡터가 1개
# x = x.T #   전치 => 행과 열을 변경 => x = x.transpose()
# #[[1,1],[2,1.1],[3,1.2], ...[10,1.3]]
# print(x.shape) #(10, 2)

# #[[[[1,2,3],[4,5,6]]]] = 1,1,2,3
# print(x+y.shape)
# # 2.model

model = Sequential()
model.add(Dense(10, input_dim=2)) 
model.add(Dense(9))
model.add(Dropout(0.2))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(1))

# input1= Input(shape=(2,)) 
# dense1= Dense(10)(input1)
# dense2= Dense(9)(dense1)
# drop1 = Dropout(0.2)(dense2)
# dense3= Dense(8,activation='relu')(drop1)
# dense4= Dense(7)(dense3)
# output1= Dense(1)(dense4)
# model= Model(inputs=input1,outputs=output1)

model.summary()
 
#  # 3.compile,fit
# model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=100, batch_size=5)

#  # 4.evauate, result
# loss = model.evaluate(x, y)
# result = model.predict([[10, 1.3]]) #   shape - 행무시, 열우선
# print("loss:", loss)
# print("[10, 1.3]:", result)

# # loss: 0.002725246362388134
# # [10, 1.3]: [[10.076436]]