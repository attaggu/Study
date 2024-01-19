import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten


#1.data
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape,y_train.shape)  #(60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape)    #(10000, 28, 28) (10000,)
# print(x_train)
# print(x_train[0])
print(y_train[0])   #5
print(np.unique(y_train,return_counts=True))    #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]

#3차원 4차원으로 변경
x_train=x_train.reshape(60000,28,28,1)  #data 내용,순서 안바뀌면 reshape 가능

# x_test=x_test.reshape(10000,28,28,1)  #아래와 같다 - 값을 모를때 적용
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
print(x_train.shape[0]) #60000
print(x_train.shape,x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)

#2.model
model = Sequential()
model.add(Conv2D(9, (2,2), input_shape=(28,28,1)))
#(N,28,28,1) ->(2,2)=커널사이즈 로 쪼개면 ->(N,27,27,1)
#Conv2D가 9면 (N,27,27,9) ====> 9 = filter 필터
# model.add(Dense(8))    #(N,27,27,8) - 데이터가 너무 커져서 평탄화 해야함
model.add(Conv2D(10,(3,3))) #(N,27,27,9)를 가져와서 (N,25,25,10)으로 던져줌 
model.add(Conv2D(15,(4,4))) #(N,25,25,10)를 가져와서 (N,22,22,15)으로 던져줌
# model.add(Conv2D(filters=15,kernel_size=(4,4)))와 같다
# shape=(batch_size, rows, coulumns,channels)
# shape=(batch_size, height, width,channels)

model.add(Flatten())    #(N,22,22,15)을 가져와서 작업
#Param => 커털X입력채널X출력채널 + 출력채널(bias) 
model.add(Dense(units=8))

model.add(Dense(7,input_shape=(8,)))
# shape=(행-batch_size,input_dim)
model.add(Dense(6))
model.add(Dense(10,activation='softmax'))    #숫자를 찾는 분류모델 / (N,27,27,10)
model.summary()




'''
#3.compile,fit
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=100,batch_size=32,verbose=1,validation_split=0.2)

#4.evaluate,predict
results=model.evaluate(x_test,y_test)
print("loss:",results[0])
print("acc:",results[1])
'''