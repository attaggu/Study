from keras.datasets import imdb
import numpy as np
import pandas as pd
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Embedding,Conv1D,Flatten,LSTM

(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=10000,)

print(x_train)
print(x_train.shape,y_train.shape)  #(25000,) (25000,)
print(x_test.shape,y_test.shape)    #(25000,) (25000,)
print(len(x_train[0]),len(x_train[1]))  #218 189
print(y_train[:30]) #[1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1 0 0 1 0 1 1 0 0 1 0] - 이진분류같음
print(np.unique(y_train,return_counts=True))   
#(array([0, 1], dtype=int64), array([12500, 12500], dtype=int64)) - 이진분류 확인
print(len(np.unique(x_train)),len(np.unique(x_test)))   #24898 24801
print(len(np.unique(y_train)),len(np.unique(y_test)))   #2 2

print(type(x_train))    #<class 'numpy.ndarray'>

print("최대 단어 길이:", max(len(i) for i in x_train))  #최대 단어 길이: 2494

print("평균 단어 길이:", sum(map(len,x_train)) / len(x_train))  #평균 단어 길이: 238.71364

x_train=pad_sequences(x_train,padding='pre',maxlen=240,truncating='pre')
x_test=pad_sequences(x_test,padding='pre',maxlen=240,truncating='pre')

print(x_train.shape,x_test.shape)   #(25000, 240) (25000, 240)
print(y_train.shape,y_test.shape)   #(25000,) (25000,)

model=Sequential()

model.add(Embedding(input_dim=10000,output_dim=15,input_length=240))
model.add(Conv1D(3,3))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam')

model.fit(x_train,y_train,epochs=10,batch_size=500,verbose=1)

result=model.evaluate(x_test,y_test)
y_predict=np.around(model.predict(x_test))

print("loss:",result)