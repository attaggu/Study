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

model.summary()
print(model.weights)

# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.47288632, -0.78825045,  1.2209238 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
# array([[-0.1723156 ,  0.5125139 ],
#        [ 0.41434443, -0.8537577 ],
#        [ 0.5188304 , -0.91461056]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
# array([[ 1.1585606],
#        [-0.4251585]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

#kernel = 가중치
print("=======================")
print(model.trainable_weights)
print("=======================")
print(len(model.weights))
print(len(model.trainable_weights))

#################
model.trainable = False # 훈련이 안됨 0이됨 - 훈련이 돼있는 모델을 가져와서 전이학습할때 가져온 모델훈련을 안시키려고 사용
#################
print(len(model.weights))
print(len(model.trainable_weights))
model.summary()
