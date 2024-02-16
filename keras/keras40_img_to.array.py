import numpy as np
import sys
import tensorflow as tf
print("tf version:",tf.__version__) #tf version: 2.9.0
print("py version:",sys.version)    #py version: 3.9.18 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator    #image를 불러와서 수치화

from tensorflow.keras.preprocessing.image import load_img #image를 불러옴
from tensorflow.keras.preprocessing.image import img_to_array #image를 수치화

path='c:/_data/image/animal/train/Cat//1.jpg'
img=load_img(path,
            #    target_size=(150,150),
               )
print(img)


print(type(img))
plt.imshow(img)
plt.show()

arr=img_to_array(img)
print(arr)
print(arr.shape)    #(281, 300, 3) 원래 크기
print(type(arr))

#차원증가
img = np.expand_dims(arr,axis=0)    #0번째 축 - 맨앞에 차원을 늘린다. (281,300,3)이면 0이면 맨앞 1이면 281,300 사이
print(img.shape)