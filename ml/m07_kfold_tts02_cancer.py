from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split,cross_val_score,KFold,StratifiedKFold,cross_val_predict
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.svm import LinearSVC, SVC

# 1. Data
x,y = load_breast_cancer(return_X_y=True)
# datasets = load_breast_cancer()
# x=datasets.data
# y=datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,random_state=1212,train_size=0.8,
                                                 stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=28)
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=28)

model=SVC()

scores = cross_val_score(model, x_train, y_train ,cv=kfold)

print("acc:",scores, "\ncv-acc:", round(np.mean(scores),4))

y_predict=cross_val_predict(model,x_test,y_test,cv=kfold)

acc=accuracy_score(y_test,y_predict)
print("acc:",acc)
