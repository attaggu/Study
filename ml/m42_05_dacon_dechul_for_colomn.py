import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터
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

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=111,train_size=0.8,stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# parameters = {
    
    
# }
parameters = {
    'objective': 'binary:logistic',  # 분류 문제인 경우 이진 분류를 위해 'binary:logistic'으로 설정합니다.
    # 'eval_metric': 'logloss',  # 모델 평가 지표로 로그 손실을 사용합니다.
    'max_depth': 6,  # 트리의 최대 깊이를 설정합니다.
    'learning_rate': 0.1,  # 학습률을 설정합니다.
    'n_estimators': 100,  # 트리의 개수를 설정합니다.
    'subsample': 0.8,  # 각 트리마다 사용될 샘플의 비율을 설정합니다.
    'colsample_bytree': 0.8,  # 각 트리마다 사용될 피처의 비율을 설정합니다.
    'reg_alpha': 0,  # L1 정규화 파라미터를 설정합니다.
    'reg_lambda': 1,  # L2 정규화 파라미터를 설정합니다.
    'random_state': 42  # 랜덤 시드를 설정합니다.
}

# 2. 모델

model = XGBClassifier()
model.set_params(early_stopping_rounds=10, **parameters)

# 3. 훈련
model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=1,
          eval_metric='mlogloss'
          )

# 4. 평가, 예측
results = model.score(x_test, y_test)
print("최종 점수 : ", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc_score : ", acc)

###############################################################
print("------------------------------------------------------------")
# print(model.feature_importances_)
feature_importances = model.feature_importances_

print(feature_importances)
# print(x_train.shape)    #(455, 30) 

num_iterations = x_train.shape[1] - 1

for i in range(num_iterations) :
    
    feature_importances =  model.feature_importances_
    min_importance_index = np.argmin(feature_importances)
    min_importance = feature_importances[min_importance_index]
    
    x_train = np.delete(x_train, min_importance_index, axis=1)
    x_test = np.delete(x_test, min_importance_index, axis=1)

    print(x_train.shape, x_test.shape)
    
    model = XGBClassifier()
    model.set_params(early_stopping_rounds=10, **parameters)
    
    model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=0,
          eval_metric='mlogloss'
          )
    
    # 4. 평가
    results = model.score(x_test, y_test)
    # print("최종 점수 : ", results)

    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    # print("acc_score : ", acc)
    print(f"acc : {acc}")
    print(f"제거된 특성 {min_importance_index}의 중요도 : {min_importance}")
    print("-----------------------------------------------------------------------")
