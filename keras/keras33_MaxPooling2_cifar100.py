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
#0.4 넘겨라
(x_train,y_train),(x_test,y_test)=cifar100.load_data()
print(x_train.shape,y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape)    #(10000, 32, 32, 3) (10000, 1)

ohe = OneHotEncoder(sparse=False)
y_train=ohe.fit_transform(y_train)
y_test=ohe.fit_transform(y_test)
# y_train_ohe=to_categorical(y_train)
# y_test_ohe=to_categorical(y_test)
x_train=x_train / 255
x_test=x_test / 255

model=Sequential()
model.add(Conv2D(2,(2,2),input_shape=(32,32,3),
                 strides=1,
                #  padding='same,'
                padding='valid'                
                 ))
model.add(MaxPooling2D())
# model.add(Dropout(0.3))
model.add(Conv2D(2,(2,2),activation='relu'))
# model.add(Dropout(0.3))
model.add(MaxPooling2D())
model.add(Conv2D(2,(2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(82,activation='relu'))
model.add(Dense(82,activation='relu'))
model.add(Dense(100,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics=['acc'])
es=EarlyStopping(monitor='val_acc',mode='auto',patience=300,
                 restore_best_weights=True)
model.fit(x_train,y_train,epochs=50,batch_size=1133,verbose=1,
          validation_split=0.18)
result=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
print("loss:",result[0])
print("acc:",result[1])
