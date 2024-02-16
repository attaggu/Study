import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN,LSTM

#RNN - 3차원 행무시

# 1 Data
datasets=([1,2,3,4,5,6,7,8,9,10])


x=np.array([[1,2,3],
            [2,3,4],
            [3,4,5],
            [4,5,6],
            [5,6,7],
            [6,7,8],
            [7,8,9]])
y=np.array([4,5,6,7,8,9,10,])

print(x.shape,y.shape)  #(7, 3) (7,)
x=x.reshape(7,3,1)
print(x.shape,y.shape)  #(7, 3, 1) (7,)


print(y.shape)


# 2 Model
model=Sequential()
# model.add(SimpleRNN(units=10,input_shape=(3,1))) units = output
model.add(LSTM(10,input_shape=(3,1)))  #(N,3,1) 에서 (3,1)을 input_shape으로 (CNN때와 똑같다)(timesteps,features)
# 3-D tensor with shape .(batch_size, timesteps, features)
model.add(Dense(17,activation='swish'))
model.add(Dense(31,activation='swish'))
model.add(Dense(17,activation='swish'))
model.add(Dense(31,activation='swish'))
model.add(Dense(1))


# 3 Compile FIt
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=1000)

# 4 Evaluate Predict
result=model.evaluate(x,y)
print("loss:",result)
x_predict=np.array([8,9,10]).reshape(1,3,1)
y_predict=model.predict(x_predict)
#(3,)데이터를 -> (1,3,1)로 변경해야함
print("???:",y_predict)
