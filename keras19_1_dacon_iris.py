from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical


path = "c://_data//dacon//iris//"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
# print(train_csv)
test_csv = pd.read_csv(path + "test.csv",index_col=0)
# print(test_csv)
submission_csv=pd.read_csv(path + "sample_submission.csv")

x=train_csv.drop(['species'],axis=1)
y=train_csv['species']

y_ohe=pd.get_dummies(y)

x_train,x_test,y_train,y_test=train_test_split(x,y_ohe,train_size=0.9,
                                               random_state=112)
print(x.shape)
print(y.shape)
model=Sequential()
model.add(Dense(1,input_dim=4))
model.add(Dense(11,activation='relu'))
model.add(Dense(11,activation='relu'))
model.add(Dense(11,activation='relu'))
model.add(Dense(11,activation='relu'))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics=['acc'])

es=EarlyStopping(monitor='val_loss',mode='min',
                 patience=1000,verbose=1,restore_best_weights=True)
hist=model.fit(x_train,y_train,epochs=4000,batch_size=100,
               validation_split=0.2,callbacks=[es])

result=model.evaluate(x_test,y_test)
print("loss:",result[0])
print("acc:",result[1])

y_predict=model.predict(x_test)
y_submit=model.predict(test_csv)
submission_csv['species']=np.argmax(y_submit,axis=1)

submission_csv.to_csv(path + "sample_submission_1.csv",index=False)
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict,axis=1)

# print(y_test)
# print(y_predict)
# print(y_test.shape,y_predict.shape)
def ACC(a,b):
    return accuracy_score(a,b)
acc=ACC(y_test,y_predict)
print("score:",acc)

