import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf

tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)

x=np.array([1,2,3,4,5])
y=np.array([1,2,3,4,5])

model= Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

print(model.weights)
model.trainable = False # 훈련이 안됨 0이됨 - 훈련이 돼있는 모델을 가져와서 전이학습할때 가져온 모델훈련을 안시키려고 사용


# model.trainable = True # 디폴트
model.compile(loss='mse', optimizer='adam')

model.fit(x,y,batch_size=1,epochs=150,verbose=0)

y_predict=model.predict(x)
print(y_predict)
