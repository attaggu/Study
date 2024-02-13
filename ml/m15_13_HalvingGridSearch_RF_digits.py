
from sklearn.model_selection import RandomizedSearchCV,train_test_split,GridSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold,cross_val_score,cross_val_predict,RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import time
import pandas as pd
import numpy as np
# datasets=load_digits()
# x=datasets.data
# y=datasets.target
x, y = load_digits(return_X_y=True)
print(x)
print(y)
print(x.shape,y.shape)  #(1797, 64) (1797,)

print(pd.value_counts(y,sort=False))    #sort = False -> 숫자순서대로 
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

# parameters = [
#     {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
#     {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
#     {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
#     {"min_samples_split": [2, 3, 5, 10]},
#     {"n_jobs": [-1, 2, 4], "min_samples_split": [2, 3, 5, 10]}
# ]
parameters = [
    {"n_jobs": [-1],"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"n_jobs": [-1],"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"n_jobs": [-1],"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1], "min_samples_split": [2, 3, 5, 10]}
]
x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,random_state=12,train_size=0.8)
splits=8
fold = KFold(n_splits=splits,shuffle=True,random_state=28)
model =HalvingGridSearchCV(RandomForestClassifier(),parameters,cv=fold,verbose=1,
                           refit=True, n_jobs=-1,factor=3,min_resources=50)
start_time=time.time()
model.fit(x_train,y_train)
end_time=time.time()
y_predict=model.predict(x_test)
print("acc.score:", accuracy_score(y_test,y_predict))
y_pred_best=model.best_estimator_.predict(x_test)

print("best_acc.score:",accuracy_score(y_test,y_pred_best))
print("time:",round(end_time-start_time,2),"s")
print("model.score:",model.score(x_test,y_test))

# GridSearchCV
# acc.score: 0.9777777777777777
# best_acc.score: 0.9777777777777777
# time: 3.23 s
# model.score: 0.9777777777777777

# RandomizedSearchCV
# acc.score: 0.9805555555555555
# best_acc.score: 0.9805555555555555
# time: 1.87 s
# model.score: 0.9805555555555555
# n_iter=20
# acc.score: 0.9805555555555555
# best_acc.score: 0.9805555555555555
# time: 3.46 s
# model.score: 0.9805555555555555

# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 50
# max_resources_: 1437
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 48
# n_resources: 50
# Fitting 8 folds for each of 48 candidates, totalling 384 fits
# ----------
# iter: 1
# n_candidates: 16
# n_resources: 150
# Fitting 8 folds for each of 16 candidates, totalling 128 fits
# ----------
# iter: 2
# n_candidates: 6
# n_resources: 450
# Fitting 8 folds for each of 6 candidates, totalling 48 fits
# ----------
# iter: 3
# n_candidates: 2
# n_resources: 1350
# Fitting 8 folds for each of 2 candidates, totalling 16 fits
# acc.score: 0.9805555555555555
# best_acc.score: 0.9805555555555555
# time: 7.17 s
# model.score: 0.9805555555555555