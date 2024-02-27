import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import keras
import random as rn
print(keras.__version__)
print(tf.__version__)
print(np.__version__)
tf.random.set_seed(123)
np.random.seed(321)
rn.seed(333)
x=np.array([1,2,3])
y=np.array([1,2,3])



model= Sequential()
model.add(Dense(5,input_dim=1,
                kernel_initializer='zeros'))
model.add(Dense(5))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100)

loss=model.evaluate(x,y,verbose=0)
print("loss:",loss)
results=model.predict([4],verbose=0)
print("4:",results)
