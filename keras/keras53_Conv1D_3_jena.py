import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,LSTM,GRU,Flatten,Conv1D
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from keras.utils import to_categorical
import os
path = 'c:/_data/kaggle/jena/'

csv = pd.read_csv(path + "jena.csv",index_col=0)
print(csv.shape)    #(420551, 14)


print(csv.isna().sum()) # 결측치X



def split_data(data,time_step, y_col,y_gap=0):
    x = []
    y = []
    num = len(data) - (time_step+y_gap) # y_col 데이터도 사용해야해서 time_step 뒤에 1이 빠짐
    for i in range(num):
        x.append(data[i : i+time_step]) 
        # y.append(data.iloc[i+time_step][y_col]) 아래와 같음
        y_row = data.iloc[i+time_step+y_gap]  #i+time_step번째 행 - x에 추가해준 바로 다음 행
        y.append(y_row[y_col])  #i+time_step번째 행에서 원하는 열의 값만 y에 추가
    return np.array(x), np.array(y)
train_size = 720
predict_gap = 144
x, y = split_data(csv, train_size, 'T (degC)',predict_gap)    
    
print("x, y :", x.shape,y.shape)    #x, y : (420383, 168, 14) (420383,)
print(x[0],y[0],sep='\n')   #확인끝 x 첫time_step까지, y 그다음 하나 확인 '\n' - 보기편한용도

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7)


model = Sequential()
# model.add(GRU(1,input_shape=x_train.shape[1:]))
model.add(Conv1D(2,kernel_size=2,input_shape=x_train.shape[1:]))
model.add(Conv1D(2,2))
model.add(Conv1D(2,2))
model.add(Conv1D(2,2))
model.add(Conv1D(2,2))
model.add(Conv1D(2,2))
model.add(Conv1D(2,2))
model.add(Conv1D(2,2))
model.add(Conv1D(2,2))
model.add(Conv1D(2,2))
model.add(Conv1D(2,2))
model.add(Conv1D(2,2))                                                 
model.add(Flatten())
model.add(Dense(3,activation='swish'))
model.add(Dense(3,activation='swish'))
model.add(Dense(1,activation='swish'))

model.compile(loss='mse',optimizer='adam')

hist=model.fit(x_train,y_train,epochs=1,batch_size=1,
            #    validation_split=0.1,
               verbose=1)

loss=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)

r2 = r2_score(y_test,y_predict)

print("loss:",loss)
print("r2:",r2)
# print(x_test.shape,y_test.shape,y_predict.shape)    #(84106, 24, 14) (84106,) (84106, 1)
y_predict=np.array(y_predict).reshape(-1,1)
y_test=np.array(y_test).reshape(-1,1)
# print(x_test.shape,y_test.shape,y_predict.shape)   #(84106, 24, 14) (84106, 1) (84106, 1) 
loss: 5.479013919830322
r2: 0.9222715962539408


# model.save(path +"jena_save")
submit = pd.DataFrame(np.array([y_test,y_predict]).reshape(-1,2),columns=['true','predict'])
submit.to_csv(path + "check_save.csv",index=False)


loss: 12.947299003601074
r2: 0.8167511418299236
