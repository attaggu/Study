import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold,cross_val_predict,RandomizedSearchCV
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
import pandas as pd

path = "C:\\_data\\dacon\\dechul\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0 )
print(train_csv.shape)  
test_csv = pd.read_csv(path + "test.csv", index_col=0 )
print(test_csv.shape) 
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv.shape)  
train_csv = train_csv[train_csv['주택소유상태'] != 'ANY']
test_csv.loc[test_csv['대출목적'] == '결혼' , '대출목적'] = '기타'
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder()
train_csv['주택소유상태'] = le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = le.fit_transform(train_csv['대출목적'])
train_csv['대출기간'] = train_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
train_csv['근로기간'] = le.fit_transform(train_csv['근로기간'])

test_csv['주택소유상태'] = le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = le.fit_transform(test_csv['대출목적'])
test_csv['대출기간'] = test_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
test_csv['근로기간'] = le.fit_transform(test_csv['근로기간'])

train_csv['대출등급'] = le.fit_transform(train_csv['대출등급'])

x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']
x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,random_state=121,stratify=y)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


splits = 3
fold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=28)
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
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=fold, verbose=1,
                     refit=True, n_jobs=-1,n_iter=30)
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
# best_score: 0.7866627895706116
# model.score: 0.5232200714463737
# acc.score: 0.5232616100357231
# best_acc.score: 0.5232200714463737
# time: 41.24 s

# RandomizedSearchCV
# best_score: 0.7731760340076711
# model.score: 0.5322339453352164
# acc.score: 0.5322339453352164
# best_acc.score: 0.5322339453352164
# time: 10.24 s
# n_iter=30
# best_score: 0.7887951924008917
# model.score: 0.5230954556783252
# acc.score: 0.5230954556783252
# best_acc.score: 0.5230123784996261
# time: 26.75 s
