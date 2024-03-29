from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.8,
                                                #  random_state=28
                                                 )
print(x_train)
print(x_test)
print(y_train)
print(y_test)

model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=2)
loss = model.evaluate(x_test,y_test)

y_predict=model.predict(x_test) # =y의 예측값 / y_test = y의 실제값
result=model.predict(x)

from sklearn.metrics import r2_score    #평가지표
r2 = r2_score(y_test,y_predict) 
print("loss:",loss)
print("R2 score:", r2)

plt.scatter(x,y)
plt.plot(x, result, color='red')
plt.show()


# 1/1 [==============================] - 0s 51ms/step ----predict
# loss: 1.7917194366455078
# R2 score: 0.8789832931866535