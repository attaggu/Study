import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPool2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

np_path='c:/_data/_save_npy//'

x=np.load(np_path+'keras39_9_x_train.npy')
y=np.load(np_path+'keras39_9_y_train.npy')
# x=x/255.
print(x.shape,y.shape)  #(2520, 150, 150, 3) (2520, 3)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=123,
                                                 train_size=0.8,stratify=y) 
train_generator=ImageDataGenerator()

augment_size=1000
randidx=np.random.randint(x_train.shape[0],size=augment_size)
x_augmented=x_train[randidx].copy()
y_augmented=y_train[randidx].copy()
x_augmented=train_generator.flow(x_augmented,y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False).next()[0]

x_train=np.concatenate((x_train,x_augmented))
y_train=np.concatenate((y_train,y_augmented))
print(x_train.shape,y_train.shape)  #(3016, 150, 150, 3) (3016, 3)

print(x_test.shape,y_test.shape)    #(504, 150, 150, 3) (504, 3)


model=Sequential()
model.add(Conv2D(11,(2,2),input_shape=(150,150,3)))
model.add(Conv2D(4,(2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(3))
model.add(Dense(3,activation='softmax'))

es=EarlyStopping(monitor='val_loss',mode='auto',patience=100,
                 restore_best_weights=True)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
hist=model.fit(x_train,y_train,epochs=30,batch_size=32,verbose=1,
               validation_split=0.15,callbacks=[es])
result=model.evaluate(x_test,y_test)

y_predict=model.predict(x_test)
argy_predict=np.argmax(y_predict,axis=1)    #제출목적이 아니면 사용 안해도 무관
# argy_predict=np.around(y_predict) 제출목적이 아니면 사용 안해도 무관

print("loss:",result[0])
print("acc:",result[1])
print(argy_predict)
# loss: 0.044129628688097
# acc: 0.9900793433189392
