from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import time #훈련시간계산

#1.data

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape) # (20640,8) (20640,)
print(datasets.feature_names)
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

print(datasets.DESCR)   #Attributes = 열, 속성, 특성, 컬럼, 차원 

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 random_state=4321)

model = Sequential()
model.add(Dense(16, input_dim=8))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train,y_train, epochs=4000, batch_size=200)
end_time = time.time()
loss=model.evaluate(x_test,y_test)

y_predict=model.predict(x_test)
result=model.predict(x)
r2 = r2_score(y_test,y_predict)
print("loss:", loss)
print("R2_score:", r2)
print("time:",round(end_time-start_time,2), "초")  #round(??? ,2) - 반올림해줌 / 2 =소수 셋째자리에서 반올림


# epochs=4000, batch_size=200, random_state=4321
# 78/78 [==============================] - 0s 540us/step - loss: 0.5607
# 162/162 [==============================] - 0s 442us/step - loss: 0.5215
# 162/162 [==============================] - 0s 398us/step
# 645/645 [==============================] - 0s 371us/step
# loss: 0.5214741826057434
# R2_score: 0.6011506496598598