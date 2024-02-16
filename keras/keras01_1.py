import tensorflow as tf # tensorflow를 땡겨오고, tf라고 줄여서 쓴다.

print(tf.__version__)   # 2.15.0
from tensorflow.keras.models import Sequential  #tf안 keras안 models안 Sequential(순차적모델)를 땡겨와주세요
from tensorflow.keras.layers import Dense   #Dense(밀집구조)
import numpy as np  #numpy를 땡겨오고 np로 읽겠다.
#1. 데이터
x = np.array([1,2,3])   #array=행렬 - np 형식에 1,2,3 
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()    #순차적 모델을 만듬
model.add(Dense(1, input_dim=1))    #그 모델을 Dense=> y=wx+b / input_dim => 1덩어리 y 한덩어리 / x=인풋 y=아웃풋(1)

#3. 컴파일, 훈련 - 최적의 weight를 구하는 과정
model.compile(loss='mse', optimizer='adam')   #mse=예측값과 실제값의 차이=loss(항상 양수) mse=제곱해서 양수로 만들겠다.
#----컴파일----
model.fit(x, y, epochs=40000) #fit=훈련 / Dense(y, input_dim=x) / epochs=훈련횟수(너무 많이 돌리면 과적합)-10번 돌려 최적W가 생성
#----훈련---- 
 
#4. 평가, 예측 - weight 값으로 나온 loss 평가
loss = model.evaluate(x, y)     #evlauate=평가 / 최적의 W가 형성된 모델상태 
print("로스 : ", loss)      #"로스 :" 수치로 프린트됨
result = model.predict([4])     #reuslt=결과값 / predict=예측
print("4의 예측값 : ", result)

# x.y => x의 y / x , y => x와 y 분리된 상태

