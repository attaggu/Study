import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

#1. Data
x, y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,random_state=123,train_size=0.8,
                                                 stratify=y)


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







