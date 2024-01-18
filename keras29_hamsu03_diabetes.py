
from keras.models import Sequential,load_model,Model
from keras.layers import Dense,Dropout,Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
from sklearn.preprocessing import RobustScaler,StandardScaler
from keras.callbacks import EarlyStopping,ModelCheckpoint
import datetime
datasets=load_diabetes()
x=datasets.data
y=datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=123)


scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)



# model=Sequential()
# model.add(Dense(1,input_dim=10))
# model.add(Dense(10))
# model.add(Dropout(0.2))
# model.add(Dense(10))
# model.add(Dropout(0.2))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dropout(0.2))
# model.add(Dense(10))
# model.add(Dense(1))

input1=Input(shape=(10,))
dense1=Dense(1)(input1)
dense2=Dense(10)(dense1)
drop1=Dropout(0.2)(dense2)
dense3=Dense(10)(drop1)
drop2=Dropout(0.2)(dense3)
dense4=Dense(10)(drop2)
dense5=Dense(10)(dense4)
drop3=Dropout(0.2)(dense5)
dense6=Dense(10)(drop3)
output1=Dense(1)(dense6)
model=Model(inputs=input1,outputs=output1)





date=datetime.datetime.now()
date=date.strftime("%m%d-%H%M")
path='../_data/_save/MCP/'
filename='{epoch:04d}-{val_loss:4f}.hdf5'
filepath="".join([path,'k26_03_diabetes_',date,'_',filename])


es=EarlyStopping(monitor='val_loss',mode='auto',patience=500,verbose=1,
                 restore_best_weights=True)
mcp=ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
                    save_best_only=True,
                    filepath=filepath)


model.compile(loss='mse', optimizer='adam')
hist=model.fit(x_train,y_train,epochs=1000,batch_size=100,validation_split=0.2,verbose=1,
               callbacks=[es,mcp])

loss=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
result=model.predict(x)


r2=r2_score(y_test,y_predict)
print("loss:",loss)
print("r2:",r2)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE:",rmse)

print(hist.history['val_loss'])

# plt.rcParams['font.family']='Malgun Gothic'
# plt.rcParams['axes.unicode_minus']=False

# plt.figure(figsize=(10,10))
# plt.plot(hist.history['loss'],c='red',label='loss',marker='.')
# plt.plot(hist.history['val_loss'],c='pink',label='val_loss',marker='.')
# plt.legend(loc='upper right')
# plt.title('당뇨병')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

# loss: 2306.89501953125
# r2: 0.6540809404569466
# RMSE: 48.0301477131255