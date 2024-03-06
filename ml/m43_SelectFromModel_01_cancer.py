import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# parameters = {
    
    
# }
parameters = {
    # 'objective': 'binary:logistic',  # 분류 문제인 경우 이진 분류를 위해 'binary:logistic'으로 설정합니다.
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
          eval_metric='logloss'
          )

# 4. 평가, 예측
results = model.score(x_test, y_test)
print("최종 점수 : ", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc_score : ", acc)

###############################################################
print("------------------------------------------------------------")

thresholds = np.sort(model.feature_importances_)
print(thresholds)
# [0.00113648 0.0024043  0.00414897 0.00488905 0.00618446 0.00679623
#  0.00761283 0.00766682 0.0080661  0.00865777 0.00885185 0.01041006
#  0.01147317 0.01154397 0.01467532 0.01513469 0.0152888  0.01660764
#  0.01803214 0.01874908 0.01899759 0.02063336 0.02084665 0.02103158
#  0.03770081 0.06312768 0.09764734 0.14844002 0.14887895 0.22436626]

from sklearn.feature_selection import SelectFromModel
# i 보다 크거나 같은 컬럼만 살아남음 ( i보다 작은 컬럼 삭제 )
for i in thresholds:
    selection = SelectFromModel(model, threshold=i,prefit=False)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(i, "\t변형된 x_train :", select_x_train.shape, "변형된 x_test :", select_x_test.shape)
    select_model = XGBClassifier()
    select_model.set_params(
    early_stopping_rounds=10,
    **parameters,
    eval_metric='logloss',
    )
    select_model.fit(select_x_train,y_train,
                 eval_set=[(select_x_train,y_train),(select_x_test,y_test)],
                 verbose=0,
                 )
    select_y_predict = select_model.predict(select_x_test)
    score = accuracy_score(y_test,select_y_predict)

    print("Trech=%.3f, n=%d, ACC: %.2f%%" %(i, select_x_train.shape[1], score*100))
    

     