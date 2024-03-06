
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
path = "c://_data//dacon//wine//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv",index_col=0)

submission_csv=pd.read_csv(path + "sample_submission.csv")

train_csv['type']=train_csv['type'].map({'white':1,'red':0}).astype(int)
test_csv['type']=test_csv['type'].map({'white':1,'red':0}).astype(int)



x=train_csv.drop(['quality'],axis=1)
y=train_csv['quality']-3
print(train_csv['quality'].value_counts().sort_index())
# 3      26
# 4     186

# 5    1788

# 6    2416

# 7     924

# 8     152
# 9       5

# print(y)














'''
group = train_csv.groupby('quality').size()
plt.bar(group.index, group.values)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Count of Labels')
plt.show()
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75]) 
    #25 - quartile_1 -1사분위
    #50 - q2 - 중위
    #75 - quartile_3 - 3사분위
    #percentile - 백분위수
    print("1사분위:", quartile_1)
    print("q2:", q2)
    print("3사분위:", quartile_3)
    iqr = quartile_3 - quartile_1
    #iqr 사분위수 범위 = q3 - q1 
    print("iqr:", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    #이상치 경계 - 이 경계를 벗어나는 데이터를 찾기 위해 np.where 함수를 사용함
    return np.where((data_out>upper_bound) | (data_out<lower_bound))    # | = or 

outlier_indices = []
for column in x.columns:
    outlier_indices.extend(outliers(x[column])[0])
x = x.drop(index=outlier_indices)
y = y.drop(index=outlier_indices)

'''
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1111,train_size=0.78,stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


parameters = {
    # 'objective': 'binary:logistic',  # 분류 문제인 경우 이진 분류를 위해 'binary:logistic'으로 설정합니다.
    # # 'eval_metric': 'mlogloss',  # 모델 평가 지표로 로그 손실을 사용합니다.
    # 'max_depth': 18,  # 트리의 최대 깊이를 설정합니다.
    # 'learning_rate': 0.3,  # 학습률을 설정합니다.
    # 'n_estimators': 59,  # 트리의 개수를 설정합니다.
    # 'subsample': 0.8,  # 각 트리마다 사용될 샘플의 비율을 설정합니다.
    # 'colsample_bytree': 0.8,  # 각 트리마다 사용될 피처의 비율을 설정합니다.
    # 'reg_alpha': 0,  # L1 정규화 파라미터를 설정합니다.
    # 'reg_lambda': 1,  # L2 정규화 파라미터를 설정합니다.
    # 'random_state': 42  # 랜덤 시드를 설정합니다.
}



model = RandomForestClassifier(random_state=661010)
# model.set_params(early_stopping_rounds=10, **parameters)

# 3. 훈련
model.fit(x_train, y_train,
        #   eval_set=[(x_train, y_train), (x_test, y_test)],
        #   verbose=1,
        #   eval_metric='mlogloss'
          )

# 4. 평가, 예측
results = model.score(x_test, y_test)
print("최종 점수 : ", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc_score : ", acc)

