from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

path = "c://_data//kaggle//bike//"
train_csv=pd.read_csv(path+"train.csv",index_col=0)
test_csv=pd.read_csv(path+"test.csv",index_col=0)
submission_csv=pd.read_csv(path+"sampleSubmission.csv")

train_csv=train_csv.dropna()
test_csv=test_csv.fillna(test_csv.mean())

x=train_csv.drop(['count','casual','registered'],axis=1)
y=train_csv['count']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=119)

model=Sequential()
model.add(Dense(1,input_dim=8,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))



from keras.optimizers import Adam
learning_rate = 0.001
model.compile(loss='mae', optimizer=Adam(learning_rate))

hist=model.fit(x_train,y_train, epochs=50, batch_size=50,
          validation_split=0.3)

loss=model.evaluate(x_test, y_test)
# y_submit = model.predict(test_csv)
# submission_csv['count']=y_submit 
# submission_csv.to_csv(path + "submission_0106.csv", index=False)
y_predict=np.round(model.predict(x_test))
r2=r2_score(y_test,y_predict)
print("lr : {0}, loss : {1}".format(learning_rate,loss))
print("lr : {0}, acc : {1}".format(learning_rate, r2))



