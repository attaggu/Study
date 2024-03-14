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
import time


#1.Data===============================
train_datagen=ImageDataGenerator(rescale=1./255,
                                #  horizontal_flip=True,
                                #  vertical_flip=True,
                                #  width_shift_range=0.1,
                                #  height_shift_range=0.1,
                                #  rotation_range=5,
                                #  zoom_range=1.2,
                                #  shear_range=0.7,
                                #  fill_mode='nearest'
                                 )

test_datagen=ImageDataGenerator(rescale=1./255)

path_train = 'c:/_data/image/brain/train//'
path_test = 'c:/_data/image/brain/test//'

xy_train=train_datagen.flow_from_directory(path_train,
                                           target_size=(200,200),
                                           batch_size=200,
                                           class_mode='binary',
                                           color_mode='grayscale',
                                           shuffle=True)
xy_test=test_datagen.flow_from_directory(path_test,
                                           target_size=(200,200),
                                           batch_size=200,
                                           class_mode='binary',
                                           color_mode='grayscale'
                                           )



print(xy_train[0][0].shape) #(160, 200, 200, 1)
print(xy_train[0][1].shape) #(160,)

print(xy_test[0][0].shape) #(120, 200, 200, 1)
print(xy_test[0][1].shape) #(120,)

np_path='c:/_data/_save_npy//'
np.save(np_path+ 'keras39_1_x_train.npy',arr=xy_train[0][0])
np.save(np_path+ 'keras39_1_y_train.npy',arr=xy_train[0][1])
np.save(np_path+ 'keras39_1_x_test.npy',arr=xy_test[0][0])
np.save(np_path+ 'keras39_1_y_test.npy',arr=xy_test[0][1])



'''
#==========================Data

# x_train=xy_train[0][0]
# y_train=xy_train[0][1]
# x_test=xy_test[0][0]
# y_test=xy_test[0][1]

model=Sequential()
model.add(Conv2D(47,(2,2),input_shape=(200,200,1)))
model.add(Conv2D(31,(2,2)))
model.add(Conv2D(27,(2,2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(31))
model.add(Dense(11))
model.add(Dropout(0.2))
model.add(Dense(31))
model.add(Dense(1,activation='sigmoid'))

model.summary()

es=EarlyStopping(monitor='val_loss',mode='auto',patience=100,
                 restore_best_weights=True)
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['acc'])
# model.fit_generator(xy_train, #업데이트가 돼서 fit이랑 fit_generator 둘 다 똑같이 가능
model.fit(xy_train, # x와 y를 안나눴을때 그냥 계산 적용 가능/ 배치사이즈를 위에서 줘야한다
                    # batch_size=15,    #적용이 안되고 위에있는 배치사이즈가 적용됨
                    steps_per_epoch=13,  #전체데이터 / batch = 160/10(위에서 적용한 배치값) = 16(디폴트) 
                                         #그 아래 배치는 데이터 손실 / 위는 에러
                    epochs=20,verbose=1,
                    validation_data=xy_test,    #validation_data로 변경 xy_test 집어넣음
                    callbacks=[es])
#UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version.
# Please use `Model.fit`, which supports generators.
result=model.evaluate(xy_test)
# y_predict=np.round(model.predict(x_test))

print("loss:",result[0])
print("acc:",result[1])

'''

xy_train = train_datagan.flow_from_directory(path_train,
                                             shuffle=True , 
                                             class_mode='categorical',
                                             color_mode= 'rgb',
                                             target_size=(300,300),                                             
                                             batch_size= 30)

x= []
y= []
for i in range(len(xy_train)) : 
    a , b = xy_train.next()
    x.append(a)
    y.append(b)

x = np.concatenate(x, axis= 0)
y = np.concatenate(y, axis= 0)