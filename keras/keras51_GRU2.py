import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM,Bidirectional,SimpleRNN,GRU
from sklearn.model_selection import train_test_split
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
print(x.shape,y.shape)  #(13, 3) (13,)
# x_predict=np.array([50,60,70])
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8)


model=Sequential()
model.add(Bidirectional(SimpleRNN(10),input_shape=(3,1)))
model.add(GRU(10,input_shape=(3,1)))
model.add(Dense(99,activation='swish'))
model.add(Dense(22,activation='swish'))
model.add(Dense(77,activation='swish'))
model.add(Dense(44,activation='swish'))
model.add(Dense(88,activation='swish'))
model.add(Dense(33,activation='swish'))
model.add(Dense(66,activation='swish'))
model.add(Dense(1,activation='swish'))

model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=200)

result=model.evaluate(x_test,y_test)
x_predict=np.array([50,60,70]).reshape(-1,3,1)
y_predict=model.predict(x_predict)

print("loss:",result)
print("???:",y_predict)

# loss: 0.01591508649289608
# ???: [[80.09449]]
