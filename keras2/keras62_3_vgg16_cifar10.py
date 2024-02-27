import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
import tensorflow as tf
from keras.datasets import cifar10
import time
tf.random.set_seed(777)
np.random.seed(777)

from keras.applications import VGG16

(x_train,y_train),(x_test,y_test)=cifar10.load_data()


start_time=time.time()
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


end_time=time.time()


model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])

model.fit(x_train,y_train,epochs=150,batch_size=200,verbose=1,validation_split=0.15)

result=model.evaluate(x_test,y_test)
# y_predict=model.predict(x_test)

print("loss:",result)

#loss: 1.0817058086395264
# acc: 0.6243000030517578

 
#  loss: 1.2157 - acc: 0.5820
# loss: [1.2157347202301025, 0.5820000171661377]