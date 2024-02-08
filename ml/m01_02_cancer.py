from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.svm import LinearSVC

# 1. Data
datasets = load_breast_cancer()

x=datasets.data
y=datasets.target


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,
                                                 random_state=2727)
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


model = LinearSVC(C=10)
model.fit(x_train,y_train)

# 4. Evaluate, Predict
loss=model.score(x_test,y_test)

# y_predict=np.around(model.predict(x_test))
y_predict=model.predict(x_test)

r2=r2_score(y_test,y_predict)

def ACC(a,b):
    return accuracy_score(a,b)
acc=ACC(y_test,y_predict)


print("ACC:",acc)
print("model.score:",loss)
print("r2:",r2)
print("acc:",acc)

'''
# plt.figure(figsize=(1,1))
# plt.plot(hist.history['loss'],c='orange',label='loss',marker='.')
# plt.plot(hist.history['val_loss'],c='red',label='val_loss',marker='.')
# plt.plot(hist.history['acc'],c='pink',label='acc',marker='.')
# plt.legend(loc='upper right')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

# ACC: 0.9298245614035088
# loss: [0.14825676381587982, 0.9298245906829834]
# r2: 0.7084398976982098
'''