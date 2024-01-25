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
np_path='c:/_data/_save_npy//'
# np.save(np_path+ 'keras39_1_x_train.npy',arr=xy_train[0][0])
# np.save(np_path+ 'keras39_1_y_train.npy',arr=xy_train[0][1])
# np.save(np_path+ 'keras39_1_x_test.npy',arr=xy_test[0][0])
# np.save(np_path+ 'keras39_1_y_test.npy',arr=xy_test[0][1])

x_train=np.load(np_path+'keras39_1_x_train.npy')
y_train=np.load(np_path+'keras39_1_y_train.npy')
x_test=np.load(np_path+'keras39_1_x_test.npy')
y_test=np.load(np_path+'keras39_1_y_test.npy')

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)



#=========================Data


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
model.fit(x_train,y_train, # x와 y를 안나눴을때 그냥 계산 적용 가능/ 배치사이즈를 위에서 줘야한다
                    batch_size=50,    #적용이 안되고 위에있는 배치사이즈가 적용됨
                    # steps_per_epoch=13,  #전체데이터 / batch = 160/10(위에서 적용한 배치값) = 16(디폴트) 
                                         #그 아래 배치는 데이터 손실 / 위는 에러
                    epochs=20,verbose=1,
                    validation_split=0.2,    #validation_data로 변경 xy_test 집어넣음
                    callbacks=[es])
#UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version.
# Please use `Model.fit`, which supports generators.
result=model.evaluate(x_test,y_test)
# y_predict=np.round(model.predict(x_test))

print("loss:",result[0])
print("acc:",result[1])

