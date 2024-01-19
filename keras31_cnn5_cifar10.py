from keras.datasets import cifar10
import numpy as np
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

print(x_train.shape,y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape)    #(10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train,return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],

ohe = OneHotEncoder(sparse=False)
y_train=ohe.fit_transform(y_train)
y_test=ohe.fit_transform(y_test)
# ohe=OneHotEncoder(sparse=False)
# y_train=y_train.reshape(-1,1)
# y_test=y_test.reshape(-1,1)
# y_train=ohe.fit_transform(y_train)
# y_test=ohe.fit_transform(y_test)
x_train=x_train/255
x_test=x_test/255

model=Sequential()
model.add(Conv2D(7,(2,2),input_shape=(32,32,3)))
model.add(Conv2D(5,(3,3)))
model.add(Conv2D(5,(2,2),activation='relu'))
model.add(Conv2D(7,(2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics=['acc'])
es=EarlyStopping(monitor='val_loss',mode='auto',patience=200,restore_best_weights=True)
model.fit(x_train,y_train,epochs=50,batch_size=1000,verbose=1,validation_split=0.15,
          callbacks=[es])
result=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)

print("loss:",result[0])
print("acc:",result[1])