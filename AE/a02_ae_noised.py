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

input_img = Input(shape=(784,))
## 인코더
### 레이어 해보고 잘나오는거 쓰면된다.
encoded = Dense(512, activation='relu')(input_img)
# encoded = Dense(32, activation='sigmoid')(input_img)
# encoded = Dense(1, activation='relu')(input_img)
# encoded = Dense(1024, activation='relu')(input_img)

## 디코더
decoded = Dense(28*28, activation='sigmoid')(encoded)
# decoded = Dense(28*28, activation='linear')(encoded)
# decoded = Dense(28*28, activation='relu')(encoded)
# decoded = Dense(28*28, activation='tanh')(encoded)

autoencoder = Model(input_img, decoded)
autoencoder.summary()

#3. 컴파일, 훈련
autoencoder.compile(optimizer='adam', loss='mse')
# autoencoder.compile(optimizer='adam', loss='mae')
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noised, x_train, epochs=30, batch_size=32, validation_split=0.2)

#4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test_noised)

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


