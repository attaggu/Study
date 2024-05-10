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

# pca ( 주성분 분석 )
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

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) = \
        plt.subplots(3, 5, figsize = (20, 7))

# 이미지 다섯 개를 무작위로 그린다.
random_images = random.sample(range(decoded_imgs.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

# 노이즈를 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(decoded_imgs[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()