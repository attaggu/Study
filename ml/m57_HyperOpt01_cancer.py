
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,LogisticRegression
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')
import time
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8,stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



search_space = {
    'learning_rate' : hp.uniform('learning_rate', 0.001,0.1),
    'max_depth' : hp.quniform('max_depth', 3,10,1),
    'num_leaves' : hp.quniform('num_leaves',24,40,1),
    'min_child_samples' : hp.quniform('min_child_samples',10,200,1),
    'min_child_weight' : hp.quniform('min_child_weight',1,50,1),
    'subsample' : hp.uniform('subsample',0.5,1),
    'colsample_bytree' : hp.uniform('colsample_bytree',0.5,1),
    'max_bin' : hp.quniform('max_bin',9,500,1),
    'reg_lambda' : hp.uniform('reg_lambda',-0.001,10),
    'reg_alpha' : hp.uniform('reg_alpha',0.01,50)
}

def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight,
              subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 100,   #epohs
        'learning_rate' : search_space['learning_rate'],    # 원래 범위 - 0.0001~0.1 소수 
        'max_depth' : int(search_space['max_depth']),    # 무조건 정수형- 레이어 층수이기 때문
        'num_leaves' : int(search_space['num_leaves']),
        'min_child_samples' : int(search_space['min_child_samples']),
        'min_child_weight' : int(search_space['min_child_weight']),
        'subsample' : max(min(search_space['subsample'],1),0),  # 0~1 사이의 값
        'colsample_bytree' : search_space['colsample_bytree'],
        'max_bin' : max(int(search_space['max_bin']), 10),   # 무조건 10 이상
        'reg_lambda' : max(search_space['reg_lambda'],0),   # 무조건 양수만
        'reg_alpha' : search_space['reg_alpha'],
    
    }
    model = XGBClassifier(**params, n_jobs =-1)
    model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
              eval_metric='logloss',
              verbose=0,
              early_stopping_rounds=50,
              )
    y_predict=model.predict(x_test)
    result = accuracy_score(y_test,y_predict)
    return result



trial_val = Trials()
n_iter= 100
start_time = time.time()
best = fmin(
    fn = xgb_hamsu,
    space = search_space,
    algo=tpe.suggest,
    max_evals=20,
    trials=trial_val,
    rstate=np.random.default_rng(seed=10)
)

end_time = time.time()

print('best:',best)
print(n_iter, '번 걸린시간 :', round(end_time-start_time,2),'초')

