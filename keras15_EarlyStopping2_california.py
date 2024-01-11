from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from keras.callbacks import EarlyStopping

datasets = fetch_california_housing()
x=datasets.data
y=datasets.target
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,random_state=950228)

model = Sequential()
model.add(Dense(16, input_dim=8))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
es=EarlyStopping(monitor='val_loss',
                 mode='min',
                 patience=15,
                 verbose=1,
                 restore_best_weights=True)

# start_time=time.time()
hist=model.fit(x_train,y_train,epochs=1000,batch_size=100,
               callbacks=[es],
               validation_split=0.3)
# end_time=time.time()

loss=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
result=model.predict(x)
r2=r2_score(y_test,y_predict)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse=RMSE(y_test,y_predict)
print("RMSE:",rmse)

print("loss:",loss)
print("r2:",r2)
# print("time",round(end_time-start_time,2),"s")
print(hist.history)
print(hist.history['loss'])
print(hist.history['val_loss'])
 
plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(13,13))
plt.plot(hist.history['loss'],c='black',label='loss',marker='.')
plt.plot(hist.history['val_loss'],c='yellow',label='val_loss',marker='.')
plt.legend(loc='upper right')
plt.title('캘리포니아')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.grid()
plt.show()
