import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,RandomForestRegressor,BaggingRegressor,VotingRegressor
import pandas as pd
from sklearn.linear_model import LogisticRegression,LinearRegression
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
path = "c:\\_data\\dacon\\ddarung\\"
train_csv=pd.read_csv(path + "train.csv",index_col=0)
test_csv=pd.read_csv(path + "test.csv",index_col=0)
submission_csv =pd.read_csv(path +"submission.csv")
train_csv=train_csv.dropna()
test_csv=test_csv.fillna(test_csv.mean())

x=train_csv.drop(['count'],axis=1)
y=train_csv['count']
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

# 2. 모델
xgb = XGBRegressor()
rf = RandomForestRegressor()
lr = LinearRegression()

model =VotingRegressor(
    estimators=[('LR', lr), ('RF', rf), ('XGB', xgb )],
    # voting='soft',
    # voting='hard',    #디폴트
    ) #최종 점수 :  0.7848596363640172
# 3. 훈련
model.fit(x_train, y_train,
        #   eval_set=[(x_train, y_train), (x_test, y_test)],
        #   verbose=1,
        #   eval_metric='logloss'
          )

# 4. 평가, 예측
results = model.score(x_test, y_test)
print("최종 점수 : ", results)

y_predict = model.predict(x_test)
###############################################################
print("------------------------------------------------------------")




