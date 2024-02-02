from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_squared_log_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler


path = "c://_data//kaggle//bike//"
train_csv=pd.read_csv(path+"train.csv",index_col=0)
test_csv=pd.read_csv(path+"test.csv",index_col=0)
submission_csv=pd.read_csv(path+"sampleSubmission.csv")

train_csv=train_csv.dropna()
test_csv=test_csv.fillna(test_csv.mean())

x=train_csv.drop(['count','casual','registered'],axis=1)
y=train_csv['count']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=819)



# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
print(x_train.shape,x_test.shape)#(8708, 8) (2178, 8)

x_train=x_train.reshape(-1,4,2,1)
x_test=x_test.reshape(-1,4,2,1)
test_csv=test_csv.to_numpy()
test_csv=test_csv.reshape(-1,4,2,1)

model=Sequential()
model.add(Conv2D(2,(2,2),input_shape=(4,2,1)))
model.add(Flatten())
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam',metrics=['mae'])

es=EarlyStopping(monitor='val_loss',mode='min',patience=1500,
                 verbose=1,restore_best_weights=True)
hist=model.fit(x_train,y_train,epochs=20,batch_size=200,
               validation_split=0.2,callbacks=[es])
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




# loss: [24363.419921875, 117.20375061035156]
# r2: 0.2572680638879674

