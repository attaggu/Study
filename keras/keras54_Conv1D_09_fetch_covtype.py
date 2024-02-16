
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,LSTM,Flatten,Conv1D
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
from sklearn.preprocessing import RobustScaler,StandardScaler
datasets = fetch_covtype()
x=datasets.data
y=datasets.target

print(x.shape,y.shape)  #(581012, 54) (581012,)
print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747

#1.
# y_ohe=to_categorical(y)
# print(y_ohe)
# print(y_ohe.shape)  #(178, 3)
#2.
# from sklearn.preprocessing import OneHotEncoder
# y=y.reshape(-1,1)
# ohe = OneHotEncoder()
# y_ohe=ohe.fit_transform(y).toarray()
# ohe=OneHotEncoder(sparse=False)
# y_ohe=ohe.fit_transform(y)
#3.
y_ohe=pd.get_dummies(y)

x_train,x_test,y_train,y_test=train_test_split(x,y_ohe,train_size=0.8,
                                               test_size=0.2,
                                               random_state=22,
                                               stratify=y_ohe)


# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)



model=Sequential()
# model.add(LSTM(1,input_shape=(54,1))) 
model.add(Conv1D(10,2,input_shape=(54,1)))
model.add(Conv1D(7,2))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
es=EarlyStopping(monitor='val_acc', mode='min',
                 patience=100,verbose=1,
                 restore_best_weights=True)
hist=model.fit(x_train,y_train,epochs=20,batch_size=1000,
               callbacks=[es],
               validation_split=0.2)
result=model.evaluate(x_test,y_test)
print("loss:",result[0])
print("acc:",result[1])

y_predict=model.predict(x_test)
print(y_predict.shape,y_test.shape) #(116203, 7) (116203, 7)
y_test=np.argmax(y_test,axis=1)
y_predict=np.argmax(y_predict,axis=1)

acc=accuracy_score(y_predict,y_test)
print("acc_score:",acc)

print(y_test)
print(y_predict)
print(y_test.shape,y_predict.shape)
