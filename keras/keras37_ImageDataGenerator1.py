import numpy as np 
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale=1./255,          #.<- 부동소수점 
    horizontal_flip=True,    #수평으로 뒤집기
    vertical_flip=True,      #수직으로 뒤집기
    width_shift_range=0.1,   #0.1만큼 평행이동
    height_shift_range=0.1,  #0.1만큼 평행이동
    rotation_range=5,        #정해진 각도만큼 이미지를 회전
    zoom_range=1.2,          #1.2배 확대
    shear_range=0.7,         #x축이나 y축을 기준으로 0~x도 사이 변환
    fill_mode='nearest',     #근처 근사값으로 적용
    )

test_datagen=ImageDataGenerator(rescale=1./255)   #train 데이터로만 훈련을 해야해 test는 안건드린다


path_train = "c://_data//image//brain//train//"
path_test = "c://_data//image//brain//test//"

xy_train=train_datagen.flow_from_directory(
    path_train,
    target_size=(200,200),  #사이즈를 맞춰 늘리거나 줄임
    batch_size=160,   #x개씩 반복으로 들어감 Iterator
    class_mode='binary',
    color_mode='grayscale',   #흑백으로 변경 / 컬러는 'rgb'
    shuffle=True,
    )   
# print(xy_train)
#Found 160 images belonging to 2 classes
#<keras.preprocessing.image.DirectoryIterator object at 0x000001D4A5424520>
#DirectoryIterator 형식으로 x와 y가 합쳐진 상태

xy_test=test_datagen.flow_from_directory(
    path_test,
    target_size=(200,200),  #사이즈를 맞춰 늘리거나 줄임
    batch_size=120,   #x개씩 반복으로 들어감 Iterator
    class_mode='binary',
    color_mode='grayscale',
    # shuffle=True,
    )
# print(xy_test)
#Found 120 images belonging to 2 classes

print(xy_train.next())
print(xy_train[0])
# # print(xy_train[16]) #error : 전체데이터/batch_size = 160/10 -> 16개인데
#                       #[16]은 17번째 값을 빼라고 하는뜻
print(xy_train[0][0])   #첫번째 배치의 x
# print(xy_train[0][1])   #첫번째 배치의 y
# print(xy_train[0][0].shape) #(10, 200, 200, 3) -> 배치사이즈를 통데이터로 10에서 160으로 늘림 -> (160, 200, 200, 3)

# print(type(xy_train))
# print(type(xy_train[0]))
# print(type(xy_train[0][0])) #0의 0번째=x
# print(type(xy_train[0][1])) #0의 1번째=y

# 통 batch가 아닐때
x= []
y= []
for i in range(len(xy_train)) : 
    a , b = xy_train.next()
    x.append(a)
    y.append(b)

x = np.concatenate(x, axis= 0)
y = np.concatenate(y, axis= 0)