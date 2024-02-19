from xgboost import XGBClassifier,XGBRFRegressor

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler

x,y=fetch_covtype(return_X_y=True)
y = y-1
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=111,train_size=0.8,stratify=y)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

splits = 5 
kFold = StratifiedKFold(n_splits=splits,shuffle=True,random_state=111)
# kFold = KFold(n_splits=splits,shuffle=True,random_state=111)

# 'n_estimators' : [100,200,300,400,500,1000] # 디폴트 100 / 1~inf / 정수 - 통상적으로 커야 좋음 / 너무 크면 과적합
# 'learning_rate' : [0.1,0.2,0.3,0.5,1,0.01,0.001] # 디폴트 0.3 / 0~1 / eta -*중요* 학습률 - 통상적으로 작으면 디테일하게해서 좋음 / 너무 작으면 오래걸리고 안좋을수도있음
# 'max_depth' : [None,2,3,4,5,6,7,8,9,10] #디폴트 6 / 0~inf / 정수
# 'gamma' : [0,1,2,3,4,5,7,10,100] #디폴트 0 / 0~inf /
# 'min_child_weight' : [0,0.01,0.001,0.1,0.5,1,5,10,100] # 디폴트 1 / 0~inf
# 'subsample' : [0,0.1,0.2,0.3,0.5,0.7,1] # 디폴트 1 / 0~1
# 'colsample_bytree' : [0,0.1,0.2,0.3,0.5,0.7,1] # 디폴트 1 / 0~1
# 'colsample_bylevel' : [0,0.1,0.2,0.3,0.5,0.7,1] # 디폴트 1 / 0~1
# 'colsample_bynode' : [0,0.1,0.2,0.3,0.5,0.7,1] # 디폴트 1 / 0~1
# 'reg_alpha' : [0,0.1,0.01,0.001,1,2,10] # 디폴트 0 / 0~inf / L1 절대값 가중치 규제 / alpha 
# 'reg_lambda' : [0,0.1,0.01,0.001,1,2,10] # 디폴트 1 / 0~inf / L2 제곱 가중치 규제 / lambda

parameters = {
    'n_estimators' : [100,200,300,400],
    'learning_rate' : [0.1,0.2,0.3,0.5,1],
    'max_depth' : [None,2,3,4,5,6,7,8,9,10],
    'gamma' : [0,1,2,3,4,5,7,10,100],
    'min_child_weight' : [0.001,0.1],
    'subsample' : [0,0.1,0.2,0.3,0.5,0.7],
    'colsample_bytree' : [0.2,0.3,0.5,0.7],
    'colsample_bylevel' : [0.2,0.3,0.5,0.7],
    'colsample_bynode' : [0.2,0.3,0.5] ,
    'reg_alpha' : [0.001,1,2] ,
    'reg_lambda' : [0.01,1,2],
}


# 2. model
xgb = XGBClassifier(random_state=111)
model = RandomizedSearchCV(xgb, parameters, cv=kFold,n_jobs=-3)

# 3. fit
model.fit(x_train,y_train)

# 4. evaluate, predict
print("최상의 매개변수:", model.best_estimator_)
print("최상의 매개변수:", model.best_params_)
print("최상의 점수:", model.best_score_)

results = model.score(x_test,y_test)
print("최종 점수:", results)
