from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense,Dropout,Flatten,Conv2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

datasets=load_boston()
x=datasets.data
y=datasets.target 

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,)
print(x_train.shape,y_train.shape)  #(404, 13) (404,)
print(x_test.shape,y_test.shape)    #(102, 13) (102,)

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1,1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1,1)

# y_trian=y_train.reshape(-1,1)
# y_test=y_test.reshape(-1,1)

# ohe=OneHotEncoder(sparse=False)
# y_train=ohe.fit_transform(y_trian)
# y_test=ohe.fit_transform(y_test)

# ohe = OneHotEncoder()
# y_train=ohe.fit_transform(y_train.reshape(-1,1)).toarray()
# y_test=ohe.fit_transform(y_test.reshape(-1,1)).toarray()



print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
print(np.unique(y_train,return_counts=True))
print(np.unique(y_test,return_counts=True))


model=Sequential()
model.add(Conv2D(10,(2,1),input_shape=(13,1,1),
                 strides=1,
                 padding='same',
))
model.add(Conv2D(22,(2,1),activation='relu'))
model.add(Conv2D(5,(2,1),activation='relu'))
model.add(Flatten())
model.add(Dense(15,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(1))
# model.summary()  
model.compile(loss='mse', optimizer='adam')
es=EarlyStopping(monitor='val_loss',mode='auto',patience=100,
                 restore_best_weights=True)
model.fit(x_train,y_train,epochs=200,batch_size=100,verbose=1,
          callbacks=[es])
result=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print("r2:",r2)
print("loss:",result)
