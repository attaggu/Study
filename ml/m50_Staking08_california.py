import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier, VotingClassifier, StackingClassifier,StackingRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from catboost import CatBoostClassifier, CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
datasets = fetch_california_housing()
x=datasets.data
y=datasets.target
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               train_size=0.8,
                                               random_state=111)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# parameters = {
    
    
# }
parameters = {
    # # 'objective': 'binary:logistic',  # 분류 문제인 경우 이진 분류를 위해 'binary:logistic'으로 설정합니다.
    # # 'eval_metric': 'logloss',  # 모델 평가 지표로 로그 손실을 사용합니다.
    # 'max_depth': 6,  # 트리의 최대 깊이를 설정합니다.
    # 'learning_rate': 0.1,  # 학습률을 설정합니다.
    # 'n_estimators': 100,  # 트리의 개수를 설정합니다.
    # 'subsample': 0.8,  # 각 트리마다 사용될 샘플의 비율을 설정합니다.
    # 'colsample_bytree': 0.8,  # 각 트리마다 사용될 피처의 비율을 설정합니다.
    # 'reg_alpha': 0,  # L1 정규화 파라미터를 설정합니다.
    # 'reg_lambda': 1,  # L2 정규화 파라미터를 설정합니다.
    # 'random_state': 42  # 랜덤 시드를 설정합니다.
}

# # 2. 모델
# xgb = XGBClassifier()
# rf = RandomForestClassifier()
# lr = LogisticRegression()

# models = [xgb, rf, lr]

# for model in models:
#     model.fit(x_train,y_train) 
#     y_predict = model.predict(x_test)
#     print(y_predict.shape)
#     # score2 = accuracy_score(y_test,y_predict)
#     # class_name = model.__class__.__name__
#     # print("{0} 정확도 : {1: 4f}".format(class_name, score2) )
    
# '''
# predicts = np.vstack([model.predict(x_test) for model in models]).T

# model12 = CatBoostClassifier()

# model12.fit(predicts,y_test)
# y_predict12=model12.predict(predicts)
# score = accuracy_score(y_test,y_predict12)

# class_name = model12.__class__.__name__
# print("{0} 정확도 : {1: 4f}".format(class_name,score ) )
# '''


# 2. 모델
xgb = XGBRegressor()
rf = RandomForestRegressor()
lr = LinearRegression()

model = StackingRegressor(
    estimators=[('LR',lr),('XGB', xgb),('RF', rf)],
    final_estimator= CatBoostRegressor(verbose=0),
    n_jobs=-1,
    # cv=5,
)

model.fit(x_train,y_train)

y_predict = model.predict(x_test)
print('model.score :',model.score(x_test,y_test))
# print('Stacking acc :', accuracy_score(y_test,y_predict))