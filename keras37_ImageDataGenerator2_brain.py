# x,y 추출해서 모델 만들기
# 성능 0.99 이상
import pandas as pd
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rotation_range=3,
                                   zoom_range=1.1,
                                   shear_range=0.2,
                                   fill_mode='nearest')
test_datagen=ImageDataGenerator(rescale=1./255)

path_train = 'c:/_data\image/brain/train//'
path_test = 'c:/_data\image/brain/test//'

xy_train=train_datagen.flow_from_directory(path_train,
                                           target_size=(200,200),
                                           batch_size=160,
                                           class_mode='binary',
                                           shuffle=True)   
print(xy_train)
#Found 160 images belonging to 2 classes
#<keras.preprocessing.image.DirectoryIterator object at 0x000001D4A5424520>
#DirectoryIterator 형식으로 x와 y가 합쳐진 상태

xy_test=test_datagen.flow_from_directory(path_test,
                                          target_size=(200,200),
                                          batch_size=120,
                                          class_mode='binary')
print(xy_test)
#Found 120 images belonging to 2 classes
# print(xy_train.next())
# print(xy_train[0])
# # print(xy_train[16]) #error : 전체데이터/batch_size = 160/10 -> 16개인데
#                       #[16]은 17번째 값을 빼라고 하는뜻
# print(xy_train[0][0])   #첫번째 배치의 x
# print(xy_train[0][1])   #첫번째 배치의 y
# print(xy_train[0][0].shape) #(10, 200, 200, 3) -> 배치사이즈를 통데이터로 10에서 160으로 늘림 -> (160, 200, 200, 3)

# print(type(xy_train))
# print(type(xy_train[0]))
# print(type(xy_train[0][0])) #0의 0번째=x
# print(type(xy_train[0][1])) #0의 1번째=y

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape,y_train.shape)  #(160, 200, 200, 3) (160,)
print(x_test.shape,y_test.shape)    #(120, 200, 200, 3) (120,)

print(pd.value_counts(y_train))
# 1.0    80
# 0.0    80
print(pd.value_counts(y_test))
# 1.0    60
# 0.0    60
print(np.unique(y_train,return_counts=True))
#(array([0., 1.], dtype=float32), array([80, 80], dtype=int64))

print(y_train.shape,y_test.shape)   #(160, 1) (120, 1)
model=Sequential()
model.add(Conv2D(10,(2,2),input_shape=(200,200,3)))
model.add(Conv2D(12,(2,2),activation='relu'))
model.add(Conv2D(12,(2,2),activation='relu'))
model.add(Conv2D(12,(2,2),activation='relu'))
model.add(Conv2D(12,(2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(11,activation='relu'))
model.add(Dense(23,activation='relu'))
model.add(Dense(11,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

es=EarlyStopping(monitor='val_acc',mode='auto',patience=250,restore_best_weights=True)
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['acc'])

model.fit(x_train,y_train,epochs=300,batch_size=50,verbose=1,validation_split=0.15,callbacks=[es])

result=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
print("loss:",result[0])
print("acc:",result[1])

# loss: 0.40339475870132446
# acc: 0.800000011920929