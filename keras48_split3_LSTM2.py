import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.model_selection import train_test_split
a = np.array(range(1,101))
x_predict = np.array(range(96,106))




#(N,4,1) -> (N,2,2) 로 변경

size = 5    # x데이터는 4개 y데이터는 1개
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        # subset = dataset[i : (i + size)]
        # aaa.append(subset)
        aaa.append(dataset[i : (i+size)])
    return np.array(aaa)


bbb = split_x(a, size)
print(bbb)
print(bbb.shape)
x = bbb[:, :-1]
y = bbb[:, -1]
print(x,y)
print(x.shape,y.shape)  #(98, 2) (98,)

x=x.reshape(-1,2,2)

print(x.shape)  #(49, 2, 2)
x_size=4
x_pre=split_x(x_predict,x_size)
print(x_pre)
# [[ 96  97  98  99]
#  [ 97  98  99 100]
#  [ 98  99 100 101]
#  [ 99 100 101 102]
#  [100 101 102 103]
#  [101 102 103 104]
#  [102 103 104 105]]


x_pre=x_pre.reshape(-1,2,2)
print(x_pre.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)


model=Sequential()
model.add(LSTM(10,input_shape=(2,2)))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=200)

result=model.evaluate(x_test,y_test)
y_predict=model.predict(x_pre)

print("loss:",result)
print("???:",y_predict)

