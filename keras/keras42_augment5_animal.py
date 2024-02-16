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
x=np.load(np_path+'keras39_3_x_train.npy')
y=np.load(np_path+'keras39_3_y_train.npy')
test=np.load(np_path+'keras39_3_test.npy')
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=123,train_size=0.8,
                                                 shuffle=True,
                                                 stratify=y)

train_generator=ImageDataGenerator(
    #   horizontal_flip=True,
    # zoom_range=0.3,
    # height_shift_range=0.2,
    # width_shift_range=0.2,
    # rotation_range=40,
    # shear_range=0.5,
    fill_mode='nearest'
)
print(x_train.shape,y_train.shape)  #(15997, 100, 100, 3) (15997,)
print(x_test.shape,y_test.shape)   #(4000, 100, 100, 3) (4000,)

augment_size=500
randidx=np.random.randint(x_train.shape[0],size=augment_size)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented=train_generator.flow(x_augmented,y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False).next()[0]

x_train=np.concatenate((x_train,x_augmented))
y_train=np.concatenate((y_train,y_augmented))
print(x_train.shape,y_train.shape)#(24997, 100, 100, 3) (24997,)


# y_train=y_train.reshape(-1,1)
# y_test=y_test.reshape(-1,1)


# ohe=OneHotEncoder(sparse=False)
# y_train=ohe.fit_transform(y_train)
# y_test=ohe.fit_transform(y_test)
print(y_train)
print(y_test)


model=Sequential()
model.add(Conv2D(2,(3,3),input_shape=(100,100,3),
                 strides=2,padding='same'))
model.add(Conv2D(1,(2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1,activation='sigmoid'))

# model.summary()

# filepath = 'c:/_data/_save/MCP/animal//'

es=EarlyStopping(monitor='val_loss',mode='auto',patience=100,
                 restore_best_weights=True)
# mcp=ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
                    # filepath=filepath)
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['acc'])
hist=model.fit(x_train,y_train,epochs=20,batch_size=32,
          verbose=1,validation_split=0.1,callbacks=[es])

result=model.evaluate(x_test,y_test)
y_predict=model.predict(test)
# y_predict=np.round(y_predict).flatten() 

y_predict=np.round(y_predict.flatten()) 
# y_predict=np.around(y_predict.reshape(-1))
print(y_predict)

print("loss:",result)

import os
filename=os.listdir('c:/_data/image/animal/test/test_animal/')
print(filename)
filename[0]=filename[0].replace(".jpg","")  # 사진파일 이름 jpg 없애서 숫자만 제목으로 남게 한다. / 0번째꺼 일단 적용해보려

len(filename)
print(len(filename))    #파일갯수 확인

for i in range(len(filename)):
    filename[i] = filename[i].replace(".jpg","")
    
print(len(filename),len(y_predict)) #둘 갯수 같나 확인

sub_df = pd.DataFrame({'Id':filename, 'Target':y_predict})
sub_df.to_csv("c:/_data/kaggle/"+"subfile_0126.csv",index=False)


# y_predict=np.round(model.predict(x_test))
# import matplotlib.image as mping
# import matplotlib.pyplot as plt

# acc=history.history['accuracy']
# vall_acc=history.hstory['val_accuracy']
# loss=history.history['loss']
# val_loss=history.history['val_loss']

# from keras.preprocessing import image


# plt.show()
