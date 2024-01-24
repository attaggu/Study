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

path_train = 'c:/_data/image/animal/train//'
# path_test = 'c:/_data/image/animal/test//'

xy_train=train_datagen.flow_from_directory(path_train,
                                           target_size=(200,200),
                                           batch_size=2000,
                                           class_mode='binary',
                                           shuffle=True)

#batch 붙이는 명렁어 =>업벤드?컨벤??
x_train=xy_train[0][0]
y_train=xy_train[0][1]
# x_test=xy_test[0][0]
# y_test=xy_test[0][1]
print(xy_train[0][0].shape)
print(xy_train[0][1].shape)


x_train,x_test,y_train,y_test=train_test_split(xy_train[0][0],xy_train[0][1],train_size=0.8,
                                               stratify=xy_train[0][1])

print(x_train.shape,y_train.shape)  #(19997, 100, 100, 3)
print(x_test.shape,y_test.shape)    #(19997,)
print(pd.value_counts(y_test))
print(np.unique(y_train,return_counts=True))


model=Sequential()
model.add(Conv2D(47,(2,2),input_shape=(200,200,3)))
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
model.fit(x_train,y_train,epochs=220,batch_size=50,verbose=1,
          validation_split=0.1,callbacks=[es])
end=time.time()

result=model.evaluate(x_test,y_test)
y_predict=np.round(model.predict(x_test))

print("loss:",result[0])
print("acc:",result[1])

print("1 next 걸린시간:", round(end-start,2))
