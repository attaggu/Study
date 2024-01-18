
from keras.models import Sequential,load_model,Model
from keras.layers import Dense,Dropout,Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_squared_log_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
import datetime

path = "c://_data//kaggle//bike//"
train_csv=pd.read_csv(path+"train.csv",index_col=0)
test_csv=pd.read_csv(path+"test.csv",index_col=0)
submission_csv=pd.read_csv(path+"sampleSubmission.csv")

train_csv=train_csv.dropna()
test_csv=test_csv.fillna(test_csv.mean())

x=train_csv.drop(['count','casual','registered'],axis=1)
y=train_csv['count']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=119)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


# model=Sequential()
# model.add(Dense(1,input_dim=8,activation='relu'))
# model.add(Dense(20,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(20,activation='relu'))
# model.add(Dense(20,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(20))
# model.add(Dropout(0.2))
# model.add(Dense(20))
# model.add(Dense(1))

input1=Input(shape=(8,),activation='relu')
dense1=Dense(1)(input1)
dense2=Dense(20,activation='relu')(dense1)
drop1=Dropout(0.2)(dense2)
dense3=Dense(20,activation='relu')(drop1)
dense4=Dense(20,activation='relu')(dense3)
drop2=Dropout(0.2)(dense4)
dense5=Dense(20)(drop2)
drop3=Dropout(0.2)(dense5)
dense6=Dense(20)(drop3)
output1=Dense(1)(dense6)
model=Model(inputs=input1,outputs=output1)








model.compile(loss='mse',optimizer='adam',metrics=['mae'])

date=datetime.datetime.now()
date=date.strftime("%m%d-%H%M")
path='../_data/_save/MCP/'
filename='{epoch:04d}-{val_loss:4f}.hdf5'
filepath="".join([path,'k26_05_kaggle_bike_',date,'_',filename])




es=EarlyStopping(monitor='val_loss',mode='min',patience=100,
                 verbose=1,restore_best_weights=True)
mcp=ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
                    save_best_only=True,
                    filepath=filepath)
hist=model.fit(x_train,y_train,epochs=1000,batch_size=50,
               validation_split=0.2,callbacks=[es,mcp])
loss=model.evaluate(x_test,y_test)

y_submit=model.predict(test_csv)
submission_csv['count']=y_submit
# submission_csv['count']=model.predict(test_csv)

submission_csv.to_csv(path+"sampleSubmission_test.csv",index=False)
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print("loss:",loss)
print("r2:",r2)
print("--:",submission_csv[submission_csv['count']<0].count())
def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))
rmse=RMSE(y_test,y_predict)
print("RMSE:",rmse)




