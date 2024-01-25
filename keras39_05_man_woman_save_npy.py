from keras.preprocessing.image import ImageDataGenerator
import numpy as np

train_datagen=ImageDataGenerator(rescale=1./255,)

path_train='c:/_data/image/people/train/'

xy_train=train_datagen.flow_from_directory(path_train,
                                           target_size=(150,150),
                                           batch_size=9999,
                                           class_mode='binary',
                                           shuffle=True)
test_datagen=ImageDataGenerator(rescale=1./255)
path_test='c:/_data/image/people/test/'

xy_test=test_datagen.flow_from_directory(path_test,
                                         target_size=(150,150),
                                         batch_size=9999,
                                         class_mode='binary')
print(xy_train[0][0].shape) #(3309, 150, 150, 3)
print(xy_train[0][1].shape) #(3309,)
print(xy_test[0][0].shape)  #(3309, 150, 150, 3)
print(xy_test[0][1].shape)  #(3309,)

np_path='c:/_data/_save_npy//'
np.save(np_path+'reas39_5_x_train.npy',arr=xy_train[0][0])
np.save(np_path+'reas39_5_y_train.npy',arr=xy_train[0][1])
np.save(np_path+'reas39_5_x_test.npy',arr=xy_test[0][0])
np.save(np_path+'reas39_5_y_test.npy',arr=xy_test[0][1])
