
from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import time
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
(x_train,y_train),(x_test,y_test) = mnist.load_data()  #pca에 y는 필요없음

start_time=time.time()
x = np.concatenate([x_train, x_test], axis=0)
y= np.concatenate([y_train,y_test],axis=0)
print(x.shape)  #(70000, 28, 28)

x =x.reshape(-1,28*28)
# y =y.reshape(-1,1)
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
