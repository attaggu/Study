from keras.datasets import cifar100
import numpy as np
import numpy as np
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

(x_train,y_train),(x_test,y_test)=cifar100.load_data()
print(x_train.shape,y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape)    #(10000, 32, 32, 3) (10000, 1)
x_trian=x_train/255.
x_test=x_test/255.
train_generator=ImageDataGenerator(
                                #    horizontal_flip=True,
                                #    zoom_range=0.3,
                                #    height_shift_range=0.2,
                                #    width_shift_range=0.2,
                                #    rotation_range=40,
                                #    shear_range=0.5,
                                #    fill_mode='nearest'
                                   )

augment_size=25000
randidx=np.random.randint(x_train.shape[0],size=augment_size)

x_augmented=x_train[randidx].copy()

y_augmented=y_train[randidx].copy()
print(x_augmented.shape)
x_augmented=x_augmented.reshape(-1,32,32,3)
print(x_augmented.shape)
x_augmented=train_generator.flow(x_augmented,y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False).next()[0]



x_train=np.concatenate((x_train,x_augmented))
y_train=np.concatenate((y_train,y_augmented))

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

ohe = OneHotEncoder(sparse=False)
y_train=ohe.fit_transform(y_train)
y_test=ohe.fit_transform(y_test)

model=Sequential()
model.add(Conv2D(2,(2,2),input_shape=(32,32,3),
                 strides=1,
                #  padding='same,'
                padding='valid'                
                 ))
model.add(MaxPooling2D())
# model.add(Dropout(0.3))
model.add(Conv2D(12,(3,3),activation='relu'))
# model.add(Dropout(0.3))
model.add(Conv2D(21,(2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(82,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(100,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics=['acc'])
es=EarlyStopping(monitor='val_acc',mode='auto',patience=300,
                 restore_best_weights=True)
model.fit(x_train,y_train,epochs=200,batch_size=1133,verbose=1,
          validation_split=0.18)
result=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
print("loss:",result[0])
print("acc:",result[1])