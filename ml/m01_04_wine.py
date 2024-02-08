from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.svm import LinearSVC

datasets = load_wine()
x=datasets.data
y=datasets.target



x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,
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


model=LinearSVC()

hist=model.fit(x_train,y_train)

result=model.score(x_test,y_test)
print("model.score:",result)

y_predict=model.predict(x_test)
# y_test = np.argmax(y_test,axis=1)
# y_predict=np.argmax(y_predict,axis=1)


acc=accuracy_score(y_predict,y_test)
print("accuray_score:",acc)