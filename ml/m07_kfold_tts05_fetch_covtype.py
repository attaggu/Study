from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold,cross_val_predict
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

x,y=fetch_covtype(return_X_y=True)
x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,random_state=121,stratify=y)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
splits = 2
# fold=KFold(n_splits=splits,shuffle=True,random_state=22)
fold = StratifiedKFold(n_splits=splits,shuffle=True,random_state=22)
model= SVC()
scores = cross_val_score(model,x_train,y_train,cv=fold)
print("acc :",scores , "\ncv-acc :", round(np.mean(scores),4))

y_predict=cross_val_predict(model,x_test,y_test,cv=fold)

acc=accuracy_score(y_test,y_predict)
print("acc:",acc)
