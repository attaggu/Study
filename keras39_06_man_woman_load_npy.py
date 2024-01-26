
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


np_path='c:/_data/_save_npy//'
x_train=np.load(np_path+'keras39_5_x_train.npy')
y_train=np.load(np_path+'keras39_5_y_train.npy')
x_test=np.load(np_path+'keras39_5_x_test.npy')
y_test=np.load(np_path+'keras39_5_y_test.npy')

print(x_train.shape,y_train.shape)#(3309, 200, 200, 3) (3309,)
print(x_test.shape,y_test.shape)#(3309, 200, 200, 3) (3309,)



model=Sequential() 
model.add(Conv2D(3,(2,2),input_shape=(200,200,3),
                 strides=2,padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(5,(2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid'))

