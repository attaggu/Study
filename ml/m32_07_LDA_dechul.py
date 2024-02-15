# sclaer , paca 후 split

import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
print(sk.__version__)   #1.1.3
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

scaler = StandardScaler()
x = scaler.fit_transform(x)


for n_classes in range(1, len(np.unique(y)) ):
    lda = LinearDiscriminantAnalysis(n_components=n_classes)
    x1 = lda.fit_transform(x, y)

    print(f"classes: {n_classes}")
    print("data shape:", x1.shape)

    # 데이터를 훈련 세트와 테스트 세트로 분할합니다.
    x_train, x_test, y_train, y_test = train_test_split(x1, y, train_size=0.8, random_state=888, shuffle=True, stratify=y)

    # 랜덤 포레스트 모델을 초기화하고 훈련합니다.
    model = RandomForestClassifier(random_state=888)
    model.fit(x_train, y_train)

    # 모델을 평가합니다.
    results = model.score(x_test, y_test)
    print('Model score:', results)
    print(lda.explained_variance_ratio_)