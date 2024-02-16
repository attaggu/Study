#06_1 카피

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,6,5,7,8,9,10])

x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])
    
    # 훈련 데이터와 평가 데이터를 따로 줌
x_val = np.array([6,7])
y_val = np.array([6,7])
    
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])


model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(x_train,y_train, epochs=1000, batch_size=400,
          verbose=2)
#verbose=0 : 침묵(출력X)
#verbose=1 : 디폴트(정보를 상세하게 출력)
#verbose=2 : 프로그래스 바 삭제(정보를 함축적으로 출력)
#verbose=3이후 : 에포만 나온다.

loss = model.evaluate(x_test,y_test)
result = model.predict([11000,7])
print("loss:", loss)
print("??:", result)
