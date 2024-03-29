import numpy as np
from sklearn.svm import SVC,SVR
from sklearn.datasets import load_boston
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import time
import pandas as pd
x,y = load_boston(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,random_state=121,
                                                 train_size=0.8)

splits = 2
fold = KFold(n_splits=splits, shuffle=True, random_state=28)
parameters = [
    {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1, 2, 4], "min_samples_split": [2, 3, 5, 10]}
]
model = GridSearchCV(RandomForestRegressor(), parameters, cv=fold, verbose=1,
                     refit=True, n_jobs=2)
start_time = time.time()
model.fit(x_train,y_train)
end_time=time.time()

print("최적의 매개변수:",model.best_estimator_)
print("최적의 파라미터:",model.best_params_)
print("best_score:",model.best_score_) 
print("model.score:", model.score(x_test,y_test)) 

# y_predict=model.predict(x_test)
# print("acc.score:", accuracy_score(y_test,y_predict))
# y_pred_best=model.best_estimator_.predict(x_test)

# print("best_acc.score:",accuracy_score(y_test,y_pred_best))
# print("time:",round(end_time-start_time,2),"s")
print(pd.DataFrame(model.cv_results_).T)
