import pandas as pd
import numpy as np
from keras.datasets  import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

train_generator=ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.3,
    height_shift_range=0.2,
    width_shift_range=0.2,
    rotation_range=40,
    shear_range=0.5,
    fill_mode='nearest'
)

augment_size = 100

# print(x_train[0].shape) #(28,28)
# plt.imshow(x_train[0])
# plt.show()

x_data=train_generator.flow(
    np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1), # x
    # np.tile - 함수를 사용하여 이미지 데이터를 반복하여 확장
    # x.train[0]을 28*28크기의 1차원배열로 바꿈
    # augment사이즈인 100번 반복 - 데이터 확장하고 변형됨
    # 이렇게 생성된 데이터를 (데이터개수,28,28,1)로 계산
    np.zeros(augment_size),     # y
    batch_size=augment_size,    #batch 통데이터
    shuffle=False,
    )   #.next()

print(x_data)
# print(x_data.shape) # tuple형태여서 에러 - flow에서 tuple형태로 반환했기 때문

print(x_data[0][0].shape)  #(100,28,28,1)
print(x_data[0][1].shape)  #(100,) y값


print(np.unique(x_data[0][1],return_counts=True))  #(array([0.]), array([100], dtype=int64))
print(x_data[0][1][0].shape)   #(array([0.]), array([100], dtype=int64))

plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i],cmap='gray')
plt.show()



