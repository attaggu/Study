

import numpy as np
from sklearn.datasets import fetch_california_housing
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
# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

# pf = PolynomialFeatures(degree=2, include_bias=False)
# x_poly = pf.fit_transform(x)



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



bayesian_params = {
    'learning_rate' : (0.001,1),
    'max_depth' : (3,10),
    'num_leaves' : (24,40),
    'min_child_samples' : (10,200),
    'min_child_weight' : (1,50),
    'subsample' : (0.5,1),
    'colsample_bytree' : (0.5,1),
    'max_bin' : (9,500),
    'reg_lambda' : (-0.001,10),
    'reg_alpha' : (0.01,50)
}

def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight,
              subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 100,   #epohs
        'learning_rate' : learning_rate,    # 원래 범위 - 0.0001~0.1 소수 
        'max_depth' : int(round(max_depth)),    # 무조건 정수형- 레이어 층수이기 때문
        'num_leaves' : int(round((num_leaves))),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample,1),0),  # 0~1 사이의 값
        'colsample_bytree' : colsample_bytree,
        'max_bin' : max(int(round(max_bin)), 10),   # 무조건 10 이상
        'reg_lambda' : max(reg_lambda,0),   # 무조건 양수만
        'reg_alpha' : reg_alpha,
    
    }
    model = XGBRegressor(**params, n_jobs =-1)
    model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
              eval_metric='logloss',
              verbose=0,
              early_stopping_rounds=50,
              )
    y_predict=model.predict(x_test)
    result = model.score(x_test,y_test)
    
    return result

bay = BayesianOptimization(
    f= xgb_hamsu,
    pbounds=bayesian_params,
    random_state=123,
)
n_iter= 100
start_time = time.time()
bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print(bay.max)
print(n_iter, '번 걸린시간 :', round(end_time-start_time,2),'초')
