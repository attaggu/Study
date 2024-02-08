from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold,cross_val_predict
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import all_estimators

x,y = load_wine(return_X_y=True)
x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,random_state=121,stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

Algorithms=all_estimators(type_filter='classifier')  #분류
# Algorithms=all_estimators(type_filter='regressor')   #회귀
print("what:", Algorithms)

print("??:",len(Algorithms))    #41개-분류모델 개수 / 55개-회귀모델 개수
n_splits=5
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=28)

for name, algorithms in Algorithms:
    try:                            #error가 뜨면 except로 넘어가게됨
        model = algorithms()
        
        scores=cross_val_score(model,x_train,y_train,cv=kfold)
        print("==========",name,"==========")
        print("acc:",scores, "\ncv-acc:", round(np.mean(scores),4))
        y_predict=cross_val_predict(model,x_test,y_test,cv=kfold)
        acc=accuracy_score(y_test,y_predict)
        print("cross:",acc)
        print(name, '의 정답률:', acc)
    except:
        print(name, '은 수정해줘야합니다.')
        continue    #



