
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler

path = "c:\\_data\\dacon\\ddarung\\"
train_csv=pd.read_csv(path + "train.csv",index_col=0)
test_csv=pd.read_csv(path + "test.csv",index_col=0)
submission_csv =pd.read_csv(path +"submission.csv")
train_csv=train_csv.dropna()
test_csv=test_csv.fillna(test_csv.mean())

x=train_csv.drop(['count'],axis=1)
y=train_csv['count']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=121)


scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


model=Sequential()
model.add(LSTM(1,input_shape=(9,1)))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
es=EarlyStopping(monitor='val_loss',mode='min',
                 patience=500,verbose=1,
                 restore_best_weights=True)
hist=model.fit(x_train,y_train,epochs=1000,batch_size=100,
               validation_split=0.2,callbacks=[es])
loss=model.evaluate(x_test,y_test)
# y_submit=model.predict(test_csv)
# submission_csv['count']=y_submit
submission_csv['count']=model.predict(test_csv)

submission_csv.to_csv(path + "submission_0116.csv",index=False)
y_predict=model.predict(x_test)

r2=r2_score(y_test,y_predict)
print("loss:",loss)
print("r2:",r2)
def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))
rmse=RMSE(y_test,y_predict)
print("RMSE:",rmse)

