
import pandas as pd
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D,LSTM,Conv1D
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

x_train=x_train.reshape(-1,100,100*3)
x_test=x_test.reshape(-1,100,100*3)


model=Sequential() 
model.add(Conv1D(5,2,input_shape=(100,100*3)))
model.add(Flatten())
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid'))


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
hist=model.fit(x_train,y_train,epochs=10,batch_size=32,verbose=1)
result=model.evaluate(x_test,y_test)

y_predict=model.predict(x_test)
y_predict=np.around(y_predict)

print("loss:",result[0])
print("acc:",result[1])


# loss: 0.5689200162887573
# acc: 0.699002742767334