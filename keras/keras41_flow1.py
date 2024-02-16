import numpy as np
import sys
import tensorflow as tf
print("tf version:",tf.__version__) #tf version: 2.9.0
print("py version:",sys.version)    #py version: 3.9.18 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator    #image를 불러와서 수치화

from tensorflow.keras.preprocessing.image import load_img #image를 불러옴
from tensorflow.keras.preprocessing.image import img_to_array #image를 수치화

path='c:/_data//image//animal//train//Cat//1.jpg'
img=load_img(path,
               target_size=(150,150),
               )
print(img)
print(type(img))    #<class 'PIL.Image.Image'>

plt.imshow(img)
# plt.show()
arr=img_to_array(img)
print(arr)
print(arr.shape)    #(281, 300, 3) 원래 크기
print(type(arr))

7#차원증        
img = np.expand_dims(arr, axis=0)    #0번째 축 - 맨앞에 차원을 늘린다. (281,300,3)이면 0이면 맨앞 1이면 281,300 사이
print(img.shape)


datagen=ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0,
    shear_range=25,
    # fill_mode='nearest'    
)
it = datagen.flow(img,  #img는 넘파이 4차원
                  batch_size=1,
                  )

# fig,ax = plt.subplots(nrows=1, ncols=5,figsize=(10,10))        #subplots-여러장을 한번에 볼때 사용 - 행1, 컬럼 5 
fig,ax = plt.subplots(nrows=5, ncols=5,figsize=(10,10))     

for row in range(5):
    for col in range(5):
        batch = it.next()
        print(batch)
        print(batch.shape)  #(1, 150, 150, 3)
        image=batch[0].astype('uint8')    #imshow 함수는 uint8 형식의 이미지를 기대하므로 형변환 추가
        # print(image)
        print("+++++++")
        print(image.shape)
        ax[row][col].imshow(image)
        ax[row][col].axis('off')
        
print(np.min(batch),np.max(batch))
plt.show()
