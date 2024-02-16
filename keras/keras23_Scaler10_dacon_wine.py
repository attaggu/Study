from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import RobustScaler,StandardScaler

path = "c://_data//dacon//wine//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv",index_col=0)
# print(test_csv)
submission_csv=pd.read_csv(path + "sample_submission.csv")

train_csv['type']=train_csv['type'].map({'white':1,'red':0}).astype(int)
test_csv['type']=test_csv['type'].map({'white':1,'red':0}).astype(int)

x=train_csv.drop(['quality'],axis=1)
y=train_csv['quality']

yo = pd.get_dummies(y)

# y_ohe = to_categorical(y)
print(yo.shape)


x_train,x_test,y_train,y_test=train_test_split(x,yo,train_size=0.9,
                                               random_state=6112, stratify= yo)
# # print(np.unique(y_ohe))


# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)




model=Sequential()
model.add(Dense(1,input_dim=12))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(7, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
es=EarlyStopping(monitor='val_loss',mode='min',
                 patience=1500,verbose=1,restore_best_weights=True)
hist=model.fit(x_train,y_train,epochs=1000,batch_size=100,
               validation_split=0.2,callbacks=[es])

result=model.evaluate(x_test,y_test)
print("loss:",result[0])
print("acc:",result[1])

y_predict=model.predict(x_test)
argy_test = np.argmax(y_test,axis=1)
argy_predict=np.argmax(y_predict,axis=1)

acc=accuracy_score(argy_test,argy_predict)
print("acc:", acc)

y_submit=np.argmax(model.predict(test_csv),axis=1)+3



submission_csv['quality']= y_submit
submission_csv.to_csv(path + "sample_submission_2.csv",index=False)


def ACC(argy_test,argy_predict):
    return accuracy_score(argy_test,argy_predict)
acc=ACC(argy_test,argy_predict)
print("score:",acc)


# plt.figure(figsize=(10,10))
# plt.plot(hist.history['acc'],c='red',label='acc',marker='.')
# plt.plot(hist.history['val_acc'],c='blue',label='val_acc',marker='.')
# plt.legend(loc='upper right')
# plt.title('wine quality')
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.grid()
# plt.show()