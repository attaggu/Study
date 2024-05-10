# 히든의 기준

import numpy as np
from keras.datasets import mnist
import tensorflow as tf
tf.random.set_seed(777)
np.random.seed(777)

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape) # 평균 0 , 표준편차 0.1인 정규분포
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
# print(x_train_noised.shape, x_test_noised.shape)    # (60000, 784) (10000, 784)
# print((np.max(x_train_noised), np.min(x_train_noised)))
# print((np.max(x_train), np.min(x_test)))

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
print((np.max(x_train_noised), np.min(x_train_noised))) # (1.0, 0.0)
print((np.max(x_test_noised), np.min(x_test_noised)))   # (1.0, 0.0)

#2. 모델
from keras.models import Sequential, Model
from keras.layers import Dense, Input

def autoencoder(hidden_layer_size) :
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(28*28,)))
    model.add(Dense(28*28, activation='sigmoid'))
    return model

# hidden_size = 713   #PCA 1.0 일때 713
hidden_size = 486   #PCA 0.999 일때
# hidden_size = 331   #PCA 0.99 일때
# hidden_size = 154   #PCA 0.95 일때

model = autoencoder(hidden_layer_size=hidden_size)

#3. 컴파일, 훈련
model.compile(optimizer='adam', loss='mse')
# autoencoder.compile(optimizer='adam', loss='mae')
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(x_train_noised, x_train, epochs=30, batch_size=32, validation_split=0.2)

#4. 평가, 예측
decoded_imgs = model.predict(x_test_noised)

import matplotlib.pyplot as plt
n = 7
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noised[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

'''
pca 주성분 분석 ( 요즘은 차원축소에 많이 사용 )
print(np.argmax(evr_cumsum>= 0.95)+1)    #154번째부터
print(np.argmax(evr_cumsum>= 0.99)+1)    #331번째부터
print(np.argmax(evr_cumsum>= 0.999)+1)   #486번째부터
print(np.argmax(evr_cumsum>= 1.0)+1)     #713번째부터 1
#argmax는 0부터 시작 evr_cumsum은 1부터 시작
'''
