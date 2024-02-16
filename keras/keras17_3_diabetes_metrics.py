from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

datasets=load_diabetes()
x=datasets.data
y=datasets.target
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,shuffle=True,random_state=12)
model=Sequential()
model.add(Dense(1,input_dim=10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
es=EarlyStopping(monitor='val_loss',
                 mode='min',
                 patience=20,
                 verbose=1,
                 restore_best_weights=True)
hist=model.fit(x_train,y_train,epochs=150,batch_size=10,
               validation_split=0.2,callbacks=[es])
loss=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
result=model.predict(x)
r2=r2_score(y_test,y_predict)
print("loss:",loss)
print("r2",r2)
def RMSE(a,b):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse=RMSE(y_test,y_predict)
print("RMSE:",rmse)

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(10,10))
plt.plot(hist.history['loss'],c='red',label='loss',marker='.')
plt.plot(hist.history['val_loss'],c='pink',label='val_loss',marker='.')
plt.legend(loc='upper right')
plt.title('당뇨')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()
