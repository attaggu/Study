#acc = 0.77 이상

from keras.datasets import cifar10
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Input
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)#(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)#(10000, 32, 32, 3) (10000, 1)
unique, count = np.unique(y_train, return_counts=True)
#[0 1 2 3 4 5 6 7 8 9] [5000 5000 5000 5000 5000 5000 5000 5000 5000 5000]
print(unique, count)
'''
Label	Description
0	airplane
1	automobile
2	bird
3	cat
4	deer
5	dog
6	frog
7	horse
8	ship
9	truck
'''
#(m,32,32,3)
print(x_train.shape[0]) # 50000
print(x_train.shape[1]) # 32
print(x_train.shape[2]) # 32

#scaling
x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

#scaler 1-1
# x_train = x_train/255.
# x_test = x_test/255.

# #scaler 1-2
# x_train = (x_train-127.5)/127.5
# x_test = (x_test-127.5)/127.5
# #scaler 2-1
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# #scaler 2-2
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)



print(x_train.shape, x_test.shape)  # (50000, 3072) (10000, 3072)


y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)

#encoding
onehot = OneHotEncoder(sparse=False)
y_train = onehot.fit_transform(y_train)
y_test = onehot.fit_transform(y_test)

#2. 모델구성
input1=Input(shape=(32*32*3,))
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



print(x_train.shape, y_train.shape)

#EarlyStop
es = EarlyStopping(monitor='val_loss', mode='min', patience=6000, restore_best_weights=True)

#3. 컴파일 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2, validation_split=0.2, callbacks=[es])



#4. 평가, 예측
results = model.evaluate(x_test, y_test)
predict =model.predict(x_test)
# acc_score =  accuracy_score(np.argmax(y_test, axis=1), predict)
# print('loss = ', results[0])
# print('acc = ', results[1])
# print('acc_score = ', acc_score)


#시각화
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,6))
# plt.plot(history.history['val_acc'], color = 'blue', label = 'val_acc', marker = '.')
# plt.plot(history.history['val_loss'], color = 'red', label = 'val_loss', marker = '.')
# plt.show()