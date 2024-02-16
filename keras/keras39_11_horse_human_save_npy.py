from keras.preprocessing.image import ImageDataGenerator
import numpy as np



train_datagen=ImageDataGenerator(rescale=1./255,)

test_datagen=ImageDataGenerator(rescale=1./255,)

path_train = 'c:/_data/image/horse_human/'

xy_train=train_datagen.flow_from_directory(path_train,
                                           target_size=(300,300),
                                           batch_size=1500,
                                           class_mode='binary',
                                           shuffle=True)
x_train=xy_train[0][0]
y_train=xy_train[0][1]

print(xy_train[0][0].shape) #(1027, 300, 300, 3)
print(xy_train[0][1].shape) #(1027, 2)

np_path='c:/_data/_save_npy//'
np.save(np_path+'keras39_11_x_train.npy',arr=xy_train[0][0])
np.save(np_path+'keras39_11_y_train.npy',arr=xy_train[0][1])
