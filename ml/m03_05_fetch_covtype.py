from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
datasets = fetch_covtype()
x=datasets.data
y=datasets.target


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,
                                               test_size=0.2)


# model=LinearSVC()
# model = Perceptron()
# model = LogisticRegression()
model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# Best KNeighborsClassifier
model.fit(x_train,y_train)
result=model.score(x_test,y_test)
print("model.score:",result)

y_predict=model.predict(x_test)

acc=accuracy_score(y_predict,y_test)
print("acc_score:",acc)
