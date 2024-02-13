
import numpy as np
from sklearn.svm import SVC,SVR
from sklearn.datasets import load_boston
from sklearn.model_selection import StratifiedKFold,cross_val_predict,RandomizedSearchCV
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import time
import pandas as pd
x,y = load_boston(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,random_state=121,
                                                 train_size=0.8)

splits = 2
fold = KFold(n_splits=splits, shuffle=True, random_state=28)
parameters = [
    {"n_jobs": [-1],"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"n_jobs": [-1],"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"n_jobs": [-1],"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1], "min_samples_split": [2, 3, 5, 10]}
]

model = HalvingGridSearchCV(RandomForestRegressor(), parameters, cv=fold, verbose=1,
                     refit=True, n_jobs=-1,
                     factor=2,min_resources=60)
start_time = time.time()
model.fit(x_train,y_train)
end_time=time.time()

print("최적의 매개변수:",model.best_estimator_)
print("최적의 파라미터:",model.best_params_)
print("best_score:",model.best_score_) 
print("model.score:", model.score(x_test,y_test)) 

y_predict=model.predict(x_test)
# print("acc.score:", accuracy_score(y_test,y_predict))
y_pred_best=model.best_estimator_.predict(x_test)

# print("best_acc.score:",accuracy_score(y_test,y_pred_best))
print("time:",round(end_time-start_time,2),"s")
print(pd.DataFrame(model.cv_results_).T)

# n_iterations: 3
# n_required_iterations: 6
# n_possible_iterations: 3
# min_resources_: 60
# max_resources_: 404
# aggressive_elimination: False
# factor: 2
# ----------
# iter: 0
# n_candidates: 48
# n_resources: 60
# Fitting 2 folds for each of 48 candidates, totalling 96 fits
# ----------
# iter: 1
# n_candidates: 24
# n_resources: 120
# Fitting 2 folds for each of 24 candidates, totalling 48 fits
# ----------
# iter: 2
# n_candidates: 12
# n_resources: 240
# Fitting 2 folds for each of 12 candidates, totalling 24 fits
# 최적의 매개변수: RandomForestRegressor(min_samples_leaf=3, min_samples_split=3, n_jobs=-1)
# 최적의 파라미터: {'min_samples_leaf': 3, 'min_samples_split': 3, 'n_jobs': -1}
# best_score: 0.767632939401921
# model.score: 0.8524902920031441
# time: 2.71 s