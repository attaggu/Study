from keras.preprocessing.image import ImageDataGenerator
import numpy as np


train_datagen=ImageDataGenerator(rescale=1./255,)

path_train='c:/_data/image/rps//'

xy_train=train_datagen.flow_from_directory(path_train,
                                           target_size=(150,150),
                                           batch_size=6000,
                                           class_mode='categorical',
                                           shuffle=True)


print(xy_train)
#Found 2520 images belonging to 3 classes.
x_train=xy_train[0][0]
y_train=xy_train[0][1]

print(x_train.shape,y_train.shape)#(2520, 150, 150, 3) (2520, 3)

np_path='c:/_data/_save_npy//'
np.save(np_path+'keras39_9_x_train.npy',arr=xy_train[0][0])
np.save(np_path+'keras39_9_y_train.npy',arr=xy_train[0][1])

