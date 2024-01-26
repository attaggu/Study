
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

start=time.time()
train_datagen=ImageDataGenerator(
                                rescale=1./255,
                                #  horizontal_flip=True,
                                #  vertical_flip=True,
                                #  width_shift_range=0.1,
                                #  height_shift_range=0.1,
                                #  rotation_range=5,
                                #  zoom_range=1.2,
                                #  shear_range=0.7,
                                #  fill_mode='nearest'
                                 )

test_datagen=ImageDataGenerator(
   rescale=1./255
                                )

path_train = 'c:/_data/image/animal/train//'
path_test = 'c:/_data/image/animal/test//'
BATCH_SIZE=int(20000)
xy_train=train_datagen.flow_from_directory(path_train,
                                           target_size=(100,100),
                                           batch_size=BATCH_SIZE,
                                           class_mode='binary',
                                           shuffle=True)

xy_test=test_datagen.flow_from_directory(path_test,
                                           target_size=(100,100),
                                           batch_size=99999,
                                           class_mode='binary',
                                        #    color_mode='grayscale'
                                           )

#batch 붙이는 명렁어 =>업벤드?컨벤??
# x_train=xy_train[0][0]
# y_train=xy_train[0][1]
# x_test=xy_test[0][0]
# y_test=xy_test[0][1]
print(xy_train[0][0].shape) #(19997, 150, 150, 3)
print(xy_train[0][1].shape) #(19997,)
print(xy_test[0][0].shape)  #(5000, 150, 150, 3)
# print(xy_test[0][1].shape)  #(5000,)

# print(xy_test[0][1])

np_path='c:/_data/_save_npy//'
np.save(np_path+'keras39_3_x_train.npy',arr=xy_train[0][0])
np.save(np_path+'keras39_3_y_train.npy',arr=xy_train[0][1])
np.save(np_path+'keras39_3_test.npy',arr=xy_test[0][0])


