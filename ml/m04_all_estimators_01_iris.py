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
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

# 1. Data
# datasets=load_iris()
# x=datasets.data
# y=datasets.target
x, y=load_iris(return_X_y=True)
print(x.shape,y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2727,stratify=y)

# 2. Model

Algorithms=all_estimators(type_filter='classifier')  #분류
# Algorithms=all_estimators(type_filter='regressor')   #회귀
print("what:", Algorithms)

print("??:",len(Algorithms))    #41개-분류모델 개수 / 55개-회귀모델 개수

for name, algorithms in Algorithms:
    try:                            #error가 뜨면 except로 넘어가게됨
        #2 Model
        model = algorithms()
        
        #3 Compile
        model.fit(x_train,y_train)
        
        acc = model.score(x_test,y_test)
        print(name, '의 정답률:', acc)
    except:
        print(name, '은 수정해줘야합니다.')
        continue    #




'''

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
