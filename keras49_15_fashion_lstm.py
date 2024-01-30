from keras.datasets import fashion_mnist
import numpy as np
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,LSTM
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping,ModelCheckpoint

(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
print(x_train.shape,y_train.shape)  #(60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape)    #(10000, 28, 28) (10000,)
print(np.unique(y_train,return_counts=True))
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)
ohe = OneHotEncoder(sparse=False)
y_train=ohe.fit_transform(y_train)
y_test=ohe.fit_transform(y_test)

x_train=x_train.reshape(-1,28,28)
x_test=x_test.reshape(-1,28,28)

model=Sequential()
model.add(LSTM(15,input_shape=(28,28)))
model.add(Dense(18,activation='relu'))
model.add(Dense(38,activation='relu'))
model.add(Dense(18,activation='relu'))
model.add(Dense(35,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics=['acc'])
es=EarlyStopping(monitor='val_acc',mode='auto',patience=300,
                 restore_best_weights=True)
model.fit(x_train,y_train,epochs=25,batch_size=1133,verbose=1,
          validation_split=0.18)

result=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
print("loss:",result[0])
print("acc:",result[1])

import matplotlib.pyplot as plt
# plt.imshow(x_train[5],'gray')
# plt.show()