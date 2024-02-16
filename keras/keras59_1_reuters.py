from keras.datasets import reuters
import numpy as np
import pandas as pd
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Embedding,Conv1D,Flatten,LSTM
(x_train,y_train),(x_test,y_test)=reuters.load_data(num_words=400,
                                                    test_split=0.2
                                                    )
# num_words = None -> 단어 전체를 씀 / x -> 가장 많이 쓴 단어 x개를 씀
# print(x_train)
# print(x_train.shape,x_test.shape)   #(8982,) (2246,)    안에 list도 있음
# print(y_train.shape,y_test.shape)   #(8982,) (2246,)
# print(y_train)
# print(y_test)
# print(np.unique(y_train))
# print(np.unique(y_test))
# # [ 3  4  3 ... 25  3 25]
# # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
# #  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
# print(len(np.unique(y_train)))  #46
# print(len(np.unique(y_test)))  #46
# print(len(np.unique(x_train)))  #8282
# print(len(np.unique(x_test)))  #2188


# print(type(x_train))    #<class 'numpy.ndarray'>    list가 안나옴
# print(type(x_train[0])) #<class 'list'>
# print(len(x_train[0]),len(x_train[1]))  #87 56 길이가 안맞음
# #길이를 맞춰주려면 최대 길이를 알아야한다

# print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)) # 2376
# #x_train 에 0번째 길이, 1번째 길이 ----반복 -> 중 max를 프린트한다

# print("뉴스기사의 평균길이 : ", sum(map(len,x_train)) / len(x_train)) # 145.5398574927633


x_train = pad_sequences(x_train, padding='pre',maxlen=150,truncating='pre')
x_test = pad_sequences(x_test, padding='pre',maxlen=150,truncating='pre')
#길이는 150, 부족한 데이터는 앞에서부터 채움, 넘치는 데이터는 앞에서부터 자름

print(len(x_train[0]),len(x_train[1]))  #150 150 맞춰짐

# y 원핫은 자유 마지막 노드 개수 46개  sparse_categorical_crossentropy

print(x_train.shape,x_test.shape)   #(8982, 150) (2246,)
print(y_train.shape,y_test.shape)   #(8982,) (2246,)


model=Sequential()

model.add(Embedding(input_dim=400,output_dim=50,input_length=150))
model.add(Conv1D(2,2))
model.add(Flatten())
model.add(Dense(46,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')

model.fit(x_train,y_train,epochs=10,batch_size=1000,verbose=1)
result=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
y_predict=np.argmax(y_predict,axis=1)

print("loss:",result)
print(y_predict)
print(y_predict.shape)

