import numpy as np
import pandas as pd
from keras.datasets  import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPool2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping,ModelCheckpoint
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
x_train=x_train/255.
x_test=x_test/255.
train_generator=ImageDataGenerator(
    # rescale=1./255,   #
    horizontal_flip=True,
    zoom_range=0.3,
    height_shift_range=0.2,
    width_shift_range=0.2,
    rotation_range=40,
    shear_range=0.5,
    fill_mode='nearest'
)

augment_size = 40000
randidx=np.random.randint(x_train.shape[0],size=augment_size) 
# np.random.randint(60000,40000) 6만개중에 4만개를 임의로 뽑아라

print(randidx)
print(randidx.shape)    #(40000,)
print(np.min(randidx),np.max(randidx))

x_augmented = x_train[randidx].copy()
print(x_augmented)
print(x_augmented.shape)    #(40000, 28, 28)

y_augmented = y_train[randidx].copy()
print(y_augmented)
print(y_augmented.shape)
#4차원으로 변경해줘야됨
x_augmented=x_augmented.reshape(-1,28,28,1) #=(x_augmented.shape[0],28,28,1)


x_augmented=train_generator.flow(
    x_augmented,y_augmented,
    batch_size=augment_size,
    shuffle=False,
    ).next()[0]    #40000개의 데이터를 변환됨
print(x_augmented)
print(x_augmented.shape)    #(40000, 28, 28, 1) - next()[0]을 줘서 x / [1]을 주면 y

x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)


print(x_train.shape)
print(x_test.shape)
x_train = np.concatenate((x_train,x_augmented))
y_train = np.concatenate((y_train,y_augmented))
print(x_train.shape,y_train.shape)  #(100000, 28, 28, 1) (100000,)
print(x_test.shape,y_test.shape)    #(10000, 28, 28, 1) (10000,)
'''
y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)
ohe=OneHotEncoder(sparse=False)
y_train=ohe.fit_transform(y_train)
y_test=ohe.fit_transform(y_test)

model=Sequential()
model.add(Conv2D(10,(3,3),input_shape=(28,28,1),
                 strides=2,padding='valid'))
model.add(MaxPool2D())
model.add(Conv2D(11,(2,2)))
model.add(Flatten())
model.add(Dense(15,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics=['acc'])
es=EarlyStopping(monitor='val_acc',mode='auto',patience=120,
                 restore_best_weights=True)
model.fit(x_train,y_train,epochs=20,batch_size=1234,verbose=1,
          validation_split=0.12)
result=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
print("loss:",result[0])
print("acc:",result[1])
'''
