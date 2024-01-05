import numpy as np
from keras.models import Sequential
from keras.layers import Dense

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])
    #train과 test를 섞어서 7:3으로 찾을 수 있는 방법

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7, #디폴트 : 0.75
                                                    test_size=0.3,  #디폴트 : 0.25
                                                    #0.7, .0.2 로 해도 돌아가지만 0.1 데이터가 손실남
                                                    #shuffle=False,  #디폴트 : True(섞임)
                                                    #random_state=42 #랜덤값을 설정, 그 값으로 고정
                                                    ) 
                                                
    #random_state = 랜덤고정값이 정해져 계속해서 똑같은 랜덤값이 나온다.


model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=2)
loss = model.evaluate(x_test,y_test)
result = model.predict(x)
print("loss:", loss)
print("???:", result)
print(x_train)
print(x_test)
print(y_train)
print(y_test)


import matplotlib.pyplot as plt
plt.scatter(x, y)   #scatter=뿌리다
plt.plot(x, result, color='red')
plt.show()