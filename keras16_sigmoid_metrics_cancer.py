from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

#분류에서는 mse를 사용하지X , binary_crossentropy 사용(이진분류에서 무조건 사용)

# 1. Data
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x=datasets.data
y=datasets.target
# print(x.shape,y.shape)  #(569,30) (569,)

# print(np.unique(y)) # y에 0,1 만 있음
# y에 0,1 갯수
print(np.unique(y,return_counts=True))
# print(np.count_nonzero(y==0))
# print(np.count_nonzero(y==1))
print(pd.value_counts(y))
# print(pd.Series(y).value_counts())
# print(pd.DataFrame(y).value_counts())

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,
                                                 random_state=2727)


# 2. Model
model=Sequential()
model.add(Dense(1,input_dim=30)) 
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))   #최종 레이어에 sigmoid =0에서 1사이 를 사용/y_predict가 정수가아닌 0~1사이 값들임

# 3. Compile, Fit
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['acc']) #accuracy=acc=정확도 - 가중치에 반영은 안됨 
                                    #mse,mae도 같이 넣을수 있음

es=EarlyStopping(monitor='val_loss',mode='min',
                 patience=20,verbose=1,
                 restore_best_weights=True)
hist=model.fit(x_train,y_train,epochs=200,batch_size=5,
              callbacks=[es],
              validation_split=0.3)

# 4. Evaluate, Predict
loss=model.evaluate(x_test,y_test)
y_predict=np.around(model.predict(x_test))
result=model.predict(x)
# r2=r2_score(y_test,y_predict)
print(y_test)
print(y_predict)


def ACC(a,b):
    return accuracy_score(a,b)
acc=ACC(y_test,y_predict)


print("ACC:",acc)
print("loss:",loss)
# print("r2:",r2)
plt.figure(figsize=(1,1))
plt.plot(hist.history['loss'],c='orange',label='loss',marker='.')
plt.plot(hist.history['val_loss'],c='red',label='val_loss',marker='.')
plt.plot(hist.history['acc'],c='pink',label='acc',marker='.')
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()
