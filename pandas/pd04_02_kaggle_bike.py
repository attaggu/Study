import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
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


x=x.astype('float32')
test_csv=test_csv.astype('float32')

# print(x.isnull().sum())
# print(test_csv.isnull().sum())

print(x.shape)

def outlier(data):
    data = pd.DataFrame(data)
    
    for label in data:
        series = data[label]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        
        iqr = q3 - q1
        upper_bound = q3 + iqr
        lower_bound = q1 - iqr
        
        series[series > upper_bound] = np.nan
        series[series < lower_bound] = np.nan
        
        # print(series.isna().sum())
        series = series.interpolate()
        data[label] = series
        
        data = data.fillna(data.mean())

    return data

x = outlier(x)
test_csv = outlier(test_csv)


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=819)



# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


model=Sequential()
model.add(Dense(1,input_dim=8,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam',metrics=['mae'])

es=EarlyStopping(monitor='val_loss',mode='min',patience=300,
                 verbose=1,restore_best_weights=True)
hist=model.fit(x_train,y_train,epochs=500,batch_size=200,
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



# loss: [22295.951171875, 110.90544891357422]
# r2: 0.2707567983237057
# --: datetime    0
# count       0
# dtype: int64
# RMSE: 149.31829757673023

# loss: [22115.94140625, 111.814697265625]
# r2: 0.2766445755811875
# --: datetime    0
# count       0
# dtype: int64
# RMSE: 148.71429152995736