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
model=Sequential()
model.add(Conv2D(11,(2,2),input_shape=(300,300,3)))
model.add(Conv2D(4,(2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(3))
model.add(Dense(2,activation='softmax'))


# print(y_train)

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