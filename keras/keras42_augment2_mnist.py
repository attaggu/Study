import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

#1.data
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train/255.
x_test=x_test/255.
trian_generator=ImageDataGenerator(fill_mode='nearest')
augment_size=20000
randidx=np.random.randint(x_train.shape[0],size=augment_size)

x_augmented=x_train[randidx].copy()
y_augmented=y_train[randidx].copy()
print(x_augmented.shape)
x_augmented=x_augmented.reshape(-1,28,28,1)
print(x_augmented.shape)

x_augmented=trian_generator.flow(x_augmented,y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False).next()[0]
print(x_augmented.shape)
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
x_train=np.concatenate((x_train,x_augmented))
y_train=np.concatenate((y_train,y_augmented))
print(x_train.shape,y_train.shape)  #(80000, 28, 28, 1) (80000,)
print(x_test.shape,y_test.shape)    #(10000, 28, 28, 1) (10000,)

y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)
ohe=OneHotEncoder(sparse=False)
y_train=ohe.fit_transform(y_train)
y_test=ohe.fit_transform(y_test)

model=Sequential()
model.add(Conv2D(10,(2,2),input_shape=(28,28,1)))
model.add(Conv2D(12,(2,2)))
model.add(Conv2D(12,(2,2)))
model.add(Flatten())
model.add(Dense(10,activation='swish'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es=EarlyStopping(monitor='val_acc',mode='min',patience=200,restore_best_weights=True)
model.fit(x_train,y_train,epochs=20,batch_size=1234,verbose=1,validation_split=0.1,callbacks=[es])
result=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
print("loss:",result[0])
print("acc:",result[1])


