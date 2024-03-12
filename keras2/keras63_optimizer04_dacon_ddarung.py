from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
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
test_csv = scaler.transform(test_csv)


model=Sequential()
model.add(Dense(1,input_dim=9))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))









from keras.optimizers import Adam
learning_rate = 1.0
model.compile(loss='mae', optimizer=Adam(learning_rate))

hist=model.fit(x_train,y_train, epochs=50, batch_size=500,
          validation_split=0.3)

loss=model.evaluate(x_test, y_test)
# y_submit = model.predict(test_csv)
# submission_csv['count']=y_submit 
# submission_csv.to_csv(path + "submission_0106.csv", index=False)
y_predict=np.round(model.predict(x_test))
acc=accuracy_score(y_test,y_predict)
print("lr : {0}, loss : {1}".format(learning_rate,loss))
print("lr : {0}, acc : {1}".format(learning_rate, acc))

# lr : 1.0, loss : 0.2002175748348236
# lr : 1.0, loss : 0.2992384908272759

# lr : 0.1, loss : 0.20375318825244904
# lr : 0.1, loss : 0.28686396677050885

# lr : 0.01, loss : 0.2002175748348236
# lr : 0.01, loss : 0.2992384908272759

# lr : 0.001, loss : 0.18642140924930573
# lr : 0.001, loss : 0.3475250951886466