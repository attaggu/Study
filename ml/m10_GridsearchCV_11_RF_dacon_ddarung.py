import numpy as np
from sklearn.svm import SVC,SVR
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import time
import pandas as pd
path = "c:\\_data\\dacon\\ddarung\\"
train_csv=pd.read_csv(path + "train.csv",index_col=0)
test_csv=pd.read_csv(path + "test.csv",index_col=0)
submission_csv =pd.read_csv(path +"submission.csv")
train_csv=train_csv.dropna()
test_csv=test_csv.fillna(test_csv.mean())

x=train_csv.drop(['count'],axis=1)
y=train_csv['count']
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
print("time:",round(end_time-start_time,2),"s")
import pandas as pd
print(pd.DataFrame(model.cv_results_).T)
