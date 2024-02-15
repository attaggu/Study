# sclaer , paca 후 split


from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
datasets = fetch_covtype()
x = datasets['data']
y = datasets.target

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