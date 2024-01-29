import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.model_selection import train_test_split
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict=np.array([50,60,70]).reshape(-1,3,1)

# x=x.reshape(-1,3,1)
print(x.shape,y.shape)  #(13, 3) (13,)
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8)


model=Sequential()
model.add(LSTM(10,return_sequences=True #LSTM을 한번 더 사용하려면 아웃풋을 3차원으로 내보내야함
               ,input_shape=(3,1),activation='swish'))
#timesteps를 Dense로 안바꾸고 그대로 보낸다
# model.add(LSTM(3,return_sequences=True))
model.add(LSTM(5))
model.add(Dense(55,activation='swish'))
model.add(Dense(3,activation='swish'))
model.add(Dense(44,activation='swish'))
model.add(Dense(23,activation='swish'))
model.add(Dense(66,activation='swish'))
model.add(Dense(31,activation='swish'))
model.add(Dense(56,activation='swish'))
model.add(Dense(1,activation='swish'))
model.summary()


model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=200)

result=model.evaluate(x_test,y_test)
y_predict=model.predict(x_predict)

print("loss:",result)
print("???:",y_predict)
