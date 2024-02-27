import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
import tensorflow as tf
from keras.datasets import cifar10
tf.random.set_seed(777)
np.random.seed(777)

from keras.applications import VGG16


vgg16=VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32,32,3),
    )

vgg16.trainable = False # 훈련을 안함 - 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))
model.summary()
