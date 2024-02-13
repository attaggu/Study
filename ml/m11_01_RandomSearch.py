import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import time

#1. Data
x, y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 shuffle=True,random_state=123,
                                                 train_size=0.8,stratify=y)

splits = 5
fold = StratifiedKFold(n_splits=splits,shuffle=True,random_state=28)
parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3, 4, 5]},  #12번
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},      #6번
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"],                      
     "gamma":[0.01, 0.001, 0.00010], "degree":[3, 4]}                   #4*6 / 24번
]   #42번

#2. Model
# model = SVC(C=1, kernel='linear',degree=3)
# model =GridSearchCV(SVC(), parameters, cv=fold, verbose=1,
#                     refit=True,  #가장 좋은놈을 빼겠다 디폴트 True
#                     n_jobs=2    #3개 코어 사용 / -1 전부 사용
#                     )

model =RandomizedSearchCV(SVC(), parameters, cv=fold, verbose=1,
                    refit=True,  #가장 좋은놈을 빼겠다 디폴트 True
                    n_jobs=-1,    #3개 코어 사용 / -1 전부 사용
                    random_state=66, #random 난수 고정
                    n_iter=20, #반복
                    )



start_time = time.time()

model.fit(x_train,y_train)

end_time = time.time()
print("최적의 매개변수:",model.best_estimator_) #가장 좋은 모델? 조합?
# 최적의 매개변수: SVC(C=1, kernel='linear')
print("최적의 파라미터:",model.best_params_)    #가장 좋은 것들 개별?
# 최적의 파라미터: {'C': 1, 'degree': 3, 'kernel': 'linear'}    
print("best_score:",model.best_score_)  #가장 좋은 fit
# best_score: 0.9916666666666667
print("model.score:", model.score(x_test,y_test))   #가장 좋은 결과?
# model.score: 0.9666666666666667

y_predict=model.predict(x_test)
print("acc.score:",accuracy_score(y_test,y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
    #SVC(C=1, kernel='linear').predict(x_test)
print("best_acc.score:",accuracy_score(y_test,y_pred_best))
print("time",round(end_time-start_time,2),"초")
import pandas as pd
print(pd.DataFrame(model.cv_results_).T)