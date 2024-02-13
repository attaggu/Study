import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold,cross_val_predict,RandomizedSearchCV
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
x,y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,random_state=121,
                                                 train_size=0.8,stratify=y)

splits = 5
fold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=28)
parameters = [
    {"n_jobs": [-1],"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},    #12
    {"n_jobs": [-1],"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},   #16
    {"n_jobs": [-1],"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},    #16
    {"n_jobs": [-1],"min_samples_split": [2, 3, 5, 10]},   #4
]

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=fold, verbose=1,
                     refit=True, n_jobs=-1,n_iter=20)
start_time = time.time()
model.fit(x_train,y_train)
end_time=time.time()

print("최적의 매개변수:",model.best_estimator_)
print("최적의 파라미터:",model.best_params_)
print("best_score:",model.best_score_) 
print("model.score:", model.score(x_test,y_test)) 

y_predict=model.predict(x_test)
print("acc.score:", accuracy_score(y_test,y_predict))
y_pred_best=model.best_estimator_.predict(x_test)

print("best_acc.score:",accuracy_score(y_test,y_pred_best))
print("time:",round(end_time-start_time,2),"s")
import pandas as pd
print(pd.DataFrame(model.cv_results_).T)

# GridSearchCV
# best_score: 0.9583333333333334
# model.score: 0.9666666666666667
# acc.score: 0.9666666666666667
# best_acc.score: 0.9666666666666667
# time: 3.17 s

# RandomizedSearchCV
# best_score: 0.9583333333333334
# model.score: 0.9666666666666667
# acc.score: 0.9666666666666667
# best_acc.score: 0.9666666666666667
# time: 1.78 s
# n_iter=20
# best_score: 0.9583333333333334
# model.score: 0.9666666666666667
# acc.score: 0.9666666666666667
# best_acc.score: 0.9666666666666667
# time: 2.08 s