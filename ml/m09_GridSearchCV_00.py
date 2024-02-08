import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
#1. Data
x, y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 shuffle=True,random_state=123,
                                                 train_size=0.8,stratify=y)

best_score = 0
for gamma in [0.01, 0.01, 0.1, 1, 10, 100]:
    for C in [0.01, 0.01, 0.1, 1, 10, 100]:
        model=SVC(gamma=gamma, C=C)
        model.fit(x_train,y_train)
        
        score=model.score(x_test,y_test)
        
        if score > best_score :
            best_score=score
            best_parameters = {'C':C, 'gamma':gamma}
            
print("최고점수: {:2f}",format(best_score) )
print("최적 매개변수: ", best_parameters)