from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler

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




model=Sequential()
model.add(Dense(11,input_dim=8))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11, activation='sigmoid'))
model.add(Dense(11, activation='sigmoid'))
model.add(Dense(11, activation='sigmoid'))
model.add(Dense(11, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['acc'])
es=EarlyStopping(monitor='val_loss',mode='min',
                 patience=1000,verbose=1,restore_best_weights=True)
hist=model.fit(x_train,y_train,epochs=2000,batch_size=100,
               validation_split=0.3,callbacks=[es])

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
