#14-1 copy
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

datasets = load_boston()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.75,
    random_state=12345)
model = Sequential()
model.add(Dense(16, input_dim=13))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor ='val_loss',
                   mode ='min',
                   patience =10,    #10번 참는다
                   verbose =1,
                   restore_best_weights=True    #가장 좋은 가중치로 돌아간다.
                   )   #earlystopping 지점을 볼수있다

####hist=history###
hist=model.fit(x_train,y_train,epochs=4000,batch_size=10,
               callbacks=[es],
               validation_split=0.2)   #callbacks로 earlystopping 넣음
###################


loss = model.evaluate(x_test,y_test)

y_predict=model.predict(x_test)
result=model.predict(x)
r2=r2_score(y_test,y_predict)
print("loss:",loss)
print("r2:",r2)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE:",rmse)


print(hist.history)    #dataset.data / datset.target 로 돼있음 ==> hist.
print(hist.history['loss'])     #loss만 따로
print(hist.history['val_loss'])     #val_loss만 따로

plt.figure(figsize=(13,13))
plt.plot(hist.history['loss'],c='red',label='loss',marker='.')
plt.plot(hist.history['val_loss'],c='blue',label='val_loss',marker='.')
plt.legend(loc='upper right')   #위치 우측 상단 라벨링
plt.title('boston loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()  #그리드 표시
plt.show()

# 14/14 [==============================] - 0s 2ms/step - loss: 24.5539 - val_loss: 30.9001
# 4/4 [==============================] - 0s 680us/step - loss: 24.7591
# 4/4 [==============================] - 0s 667us/step
# 16/16 [==============================] - 0s 467us/step
# loss: 24.75914764404297
# r2: 0.6149876572231111