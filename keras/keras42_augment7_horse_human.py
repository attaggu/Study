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

x=np.load(np_path+'keras39_7_x_train.npy')
y=np.load(np_path+'keras39_7_y_train.npy')

# print(x.shape,y.shape)  #(1027, 300, 300, 3) (1027,)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=123,
                                                 train_size=0.8,stratify=y) 
# x_train=x_train/255.
# x_test=x_test/255.
train_generator=ImageDataGenerator(fill_mode='nearest')

print(x_train.shape,y_train.shape)  #(821, 300, 300, 3) (821, 2)
print(x_test.shape,y_test.shape)    #(206, 300, 300, 3) (206, 2)

augment_size=1000
randidx=np.random.randint(x_train.shape[0],size=augment_size)
x_augmented=x_train[randidx].copy()
y_augmented=y_train[randidx].copy()
print(x_augmented.shape,y_augmented.shape)

x_augmented=train_generator.flow(x_augmented,y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False).next()[0]
x_train=np.concatenate((x_train,x_augmented))
y_train=np.concatenate((y_train,y_augmented))

print(x_train.shape,y_train.shape)  #(1821, 300, 300, 3) (1821, 2)




model=Sequential()
model.add(Conv2D(27,(2,2),input_shape=(300,300,3)))
model.add(Conv2D(21,(2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(25))
model.add(Dense(2,activation='softmax'))

filepath='c:/_data/_save/MCP/horse_human'
es=EarlyStopping(monitor='val_loss',mode='auto',patience=100,
                 restore_best_weights=True)
mcp=ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
                    filepath=filepath)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
hist=model.fit(x_train,y_train,epochs=30,batch_size=32,verbose=1,
               validation_split=0.15,callbacks=[es,mcp])
result=model.evaluate(x_test,y_test)


y_predict=model.predict(x_test)


# y_predict=y_predict.flatten()
argy_predict=np.argmax(y_predict,axis=1)

print("loss:",result[0])
print("acc:",result[1])
print(argy_predict)
# loss: 0.015083509497344494
# acc: 0.9951456189155579