import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression 
#LogisticRegression - 분류
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# 1. Data
# datasets=load_iris()
# x=datasets.data
# y=datasets.target
x, y=load_iris(return_X_y=True)
print(x.shape,y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=28,stratify=y)

# 2. Model
# model = LinearSVC()
# model = Perceptron()
# model = LogisticRegression()
model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

#Best - KNeighborsClassifier

# 3. Compile, Fit
model.fit(x_train,y_train)

# 4. Evaluate, Predict
result= model.score(x_test,y_test)
print("model.score:",result)

y_predict=model.predict(x_test)
print(y_predict)

acc=accuracy_score(y_test,y_predict)
print("acc", acc)

'''
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict,axis=1)
'''
