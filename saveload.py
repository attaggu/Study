import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense , Conv2D ,Flatten , MaxPooling2D , Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import time


#1 데이터
np_path = "C:\\_data\\_save_npy\\"

x = np.load(np_path + 'keras39_3_x_train.npy')
y = np.load(np_path + 'keras39_3_y_train.npy')
test = 'C:\_data\image\CatDog\Test'



x_train , x_test, y_train , y_test = train_test_split(
    x, y,random_state= 3702 , shuffle= True,
    stratify=y)






print(x_train.shape)




#2 모델구성
model = Sequential()
model.add(Conv2D(64,(2,2),input_shape = (500,500,3) , strides=1 , activation='relu' ))
model.add(MaxPooling2D())
model.add(Conv2D(32,(2,2), activation='relu' ))
model.add(Conv2D(28,(2,2), activation='relu' ))
model.add(Conv2D(24,(2,2), activation='relu' ))
model.add(Conv2D(20,(2,2), activation='relu' ))
model.add(BatchNormalization())
model.add(Conv2D(16,(2,2), activation='relu' ))
model.add(Conv2D(12,(2,2), activation= 'relu' ))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(80,activation='relu'))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# model.summary()

#3 컴파일, 훈련
filepath = "C:\_data\_save\MCP\_k39\CatDog"

from keras.callbacks import EarlyStopping,ModelCheckpoint
import time

es = EarlyStopping(monitor='val_loss' , mode = 'auto' , patience= 100 , restore_best_weights=True , verbose= 1  )
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath= filepath)


model.compile(loss= 'binary_crossentropy' , optimizer='adam' , metrics=['acc'] )
hist = model.fit(x_train,y_train, epochs = 1000 , batch_size= 50 , validation_split= 0.2, verbose= 2 ,callbacks=[es, mcp])


#4 평가, 예측
result = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
y_prediect = np.around(y_predict.reshape(-1))

print('loss',result)
end_time = time.time()
print('걸린시간 : ' , round(end_time - start_time,2), "초" )


import os
path = 'C:\\_data\\image\\CatDog\\'

forder_dir = path+"test\\test"
id_list = os.listdir(forder_dir)
for i, id in enumerate(id_list):
    id_list[i] = int(id.split('.')[0])

for id in id_list:
    print(id)

y_submit = pd.DataFrame({'id':id_list,'Target':y_predict})
print(y_submit)
y_submit.to_csv(path+f"submit\\acc_{result[1]:.6f}.csv",index=False)







import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['val_acc'], c = 'pink', label = 'val_acc', marker = '.')
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss', marker = '.')

plt.title('cat_dog')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.grid
plt.show()

