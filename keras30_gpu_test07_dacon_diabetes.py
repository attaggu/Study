
from keras.models import Sequential,load_model,Model
from keras.layers import Dense,Dropout,Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
import datetime
import time
path = "c://_data//dacon//diabetes//"
train_csv = pd.read_csv(path + "train.csv",index_col=0)
print(train_csv.shape)
test_csv = pd.read_csv(path + "test.csv",index_col=0)
print(test_csv.shape)
submission_csv=pd.read_csv(path + "sample_submission.csv")

train_csv=train_csv.dropna()

x = train_csv.drop(['Outcome'],axis=1)
print(train_csv.shape)
y = train_csv['Outcome']
print(test_csv.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,
                                               random_state=666)


# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)




# model=Sequential()
# model.add(Dense(11,input_dim=8))
# model.add(Dense(11))
# model.add(Dense(11))
# model.add(Dense(11))
# model.add(Dense(11))
# model.add(Dropout(0.2))
# model.add(Dense(11, activation='sigmoid'))
# model.add(Dense(11, activation='sigmoid'))
# model.add(Dense(11, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))


input1=Input(shape=(8,))
dense1=Dense(11)(input1)
dense2=Dense(11)(dense1)
dense3=Dense(11)(dense2)
dense4=Dense(11)(dense3)
dense5=Dense(11)(dense4)
drop1=Dropout(0.2)(dense5)
dense6=Dense(11,activation='sigmoid')(drop1)
dense7=Dense(11,activation='sigmoid')(dense6)
dense8=Dense(11,activation='sigmoid')(dense7)
output1=Dense(1,activation='sigmoid')(dense8)
model=Model(inputs=input1,outputs=output1)


date=datetime.datetime.now()
date=date.strftime("%m%d-%H%M")
path='../_data/_save/MCP/'
filename='{epoch:04d}-{val_loss:4f}.hdf5'
filepath="".join([path,'k26_07_dacon_diabetes_',date,'_',filename])




model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['acc'])
es=EarlyStopping(monitor='val_loss',mode='min',
                 patience=1000,verbose=1,restore_best_weights=True)
mcp=ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
                    filepath=filepath)

start_time=time.time()
hist=model.fit(x_train,y_train,epochs=2000,batch_size=100,
               validation_split=0.3,callbacks=[es,mcp])
end_time=time.time()

loss=model.evaluate(x_test,y_test)
y_predict=np.round(model.predict(x_test))
# y_submit=np.round(model.predict(test_csv))
# submission_csv['Outcome']=y_submit
submission_csv['Outcome']=np.round(model.predict(test_csv))

submission_csv.to_csv(path + "sample_submission_1.csv",index=False)
# y_predict=np.round(model.predict(x_test))


def ACC(a,b):
    return accuracy_score(a,b)
acc=ACC(y_test,y_predict)

print("ACC:",acc)
print("time:",round(end_time-start_time,2),"s")