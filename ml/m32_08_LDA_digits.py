# sclaer , paca 후 split


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
i
print(sk.__version__)   #1.1.3

datasets = load_digits()
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

# lda = LinearDiscriminantAnalysis()    #라벨 개수에 영향을탐 - min(n_features, n_classes - 1) ->4, 3-1 이여서 2까지만 쓸수있음 / ()=디폴트
# x = lda.fit_transform(x,y)

# print(x.shape,y.shape)  #(150, 4) (150,)
# x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=888,shuffle=True,stratify=y)

# model = RandomForestClassifier(random_state=888)

# model.fit(x_train,y_train)
# results = model.score(x_test,y_test)
# print('model.score:', results)

# print(lda.explained_variance_ratio_)
