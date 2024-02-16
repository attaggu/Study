import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,6,5,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,6,5,7])
    
    # 훈련 데이터와 평가 데이터를 따로 줌
    
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])


model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(x_train,y_train, epochs=1000, batch_size=5)
loss = model.evaluate(x_test,y_test)
result = model.predict([11000,7])
print("loss:", loss)
print("??:", result)

# Epoch 1000/1000
# 2/2 [==============================] - 0s 0s/step - loss: 0.2976
# 1/1 [==============================] - 0s 74ms/step - loss: 0.1882
# 1/1 [==============================] - 0s 67ms/step
# loss: 0.18823449313640594
# ??: [[1.0306027e+04]
#  [6.6954117e+00]]
