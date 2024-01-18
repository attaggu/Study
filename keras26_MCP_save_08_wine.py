
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils import to_categorical
from keras.models import Sequential,load_model
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
import datetime

datasets = load_wine()
x=datasets.data
y=datasets.target
print(x.shape,y.shape)  #(178, 13) (178,)
print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48

#1.
# y_ohe=to_categorical(y)
# print(y_ohe)
# print(y_ohe.shape)  #(178, 3)
#2.
# from sklearn.preprocessing import OneHotEncoder
# y=y.reshape(-1,1)
# ohe = OneHotEncoder()
# y_ohe=ohe.fit_transform(y).toarray()
# ohe=OneHotEncoder(sparse=False)
# y_ohe=ohe.fit_transform(y)
#3.
y_ohe=pd.get_dummies(y)


x_train,x_test,y_train,y_test=train_test_split(x,y_ohe,train_size=0.8,
                                               test_size=0.2,
                                               random_state=22,
                                               stratify=y)


scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)



model=Sequential()
model.add(Dense(1,input_dim=13)) 
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

date=datetime.datetime.now()
date=date.strftime("%m%d-%H%M")
path='../_data/_save/MCP/'
filename='{epoch:04d}-{val_loss:4f}.hdf5'
filepath="".join([path,'k26_08_wine_',date,'_',filename])



model.compile(loss='categorical_crossentropy',optimizer='adam',
             metrics=['acc']
             )
es=EarlyStopping(monitor='val_acc',mode='min',
                 patience=20,verbose=1,
                 restore_best_weights=True)
mcp=ModelCheckpoint(monitor='val_loss',mode='min',verbose=1,save_best_only=True,
                    filepath=filepath)

hist=model.fit(x_train,y_train,epochs=1000,batch_size=10,
               callbacks=[es,mcp],
               validation_split=0.2)

result=model.evaluate(x_test,y_test)
print("loss:",result[0])
print("acc:",result[1])

y_predict=model.predict(x_test)
# print(y_predict)
print(y_predict.shape,y_test.shape) #(36, 3) (36, 3)
y_test = np.argmax(y_test,axis=1)
y_predict=np.argmax(y_predict,axis=1)

print(y_test)
print(y_predict)
print(y_test.shape,y_predict.shape)

acc=accuracy_score(y_predict,y_test)
print("accuray_score:",acc)