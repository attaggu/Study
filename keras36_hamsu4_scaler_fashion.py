# 0.99 이상 맹글기
from keras.datasets import fashion_mnist
import numpy as np
import pandas as pd
from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout,Input
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,StandardScaler

# 1 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(y_test.shape, y_test.shape)   # (10000,) (10000,)


#scaler 1-1
# x_train = x_train/255.
# x_test = x_test/255.

# #scaler 1-2
# x_train = (x_train-127.5)/127.5
# x_test = (x_test-127.5)/127.5

# #scaler 2-1
# scaler=MinMaxScaler()
#scaler.fit(x_train)
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.fit_transform(x_test)

# # #scaler 2-2
# scaler=StandardScaler()
# # scaler.fit(x_train)
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.fit_transform(x_test)




print(x_train)
print(x_train[0])   
print(x_train[1])   
print(np.unique(y_train, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],
print(pd.value_counts(y_test))

# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)

one_hot = OneHotEncoder()
y_train = one_hot.fit_transform(y_train).toarray()
y_test = one_hot.transform(y_test).toarray()

#2. 모델


input1=Input(shape=(28*28,))
dense1=Dense(100,activation='relu')(input1)
drop1=Dropout(0.3)(dense1)
dense2=Dense(10,activation='relu')(drop1)
drop2=Dropout(0.3)(dense2)
dense3=Dense(100,activation='relu')(drop2)
drop3=Dropout(0.3)(dense3)
dense4=Dense(100,activation='relu')(drop3)
dense5=Dense(60,activation='relu')(dense4)
drop4=Dropout(0.3)(dense5)
dense6=Dense(30,activation='relu')(drop4)
output1=Dense(10,activation='softmax')(dense6)
model=Model(inputs=input1,outputs=output1)



model.summary()# import matplotlib.pyplot as plt
# plt.imshow(x_train[1], 'gray')
# plt.show()

#3 컴파일, 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=2000, verbose= 1, epochs= 40, validation_split=0.2 )

#4.평가, 예측
results = model.evaluate(x_test, y_test)
print('loss = ', results[0])
print('acc = ', results[1])

# y_test_armg =  np.argmax(y_test, axis=1)
# predict = np.argmax(model.predict(x_test),axis=1)
# print(predict)

# loss =  2.8183517456054688
# acc =  0.8798999786376953

# loss =  2.23126220703125
# acc =  0.8815000057220459
# 313/313 [==============================] - 0s 800us/step
# [9 2 1 ... 8 1 5]