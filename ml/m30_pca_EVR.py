# sclaer , paca 후 split

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
print(sk.__version__)   #1.1.3

datasets = load_breast_cancer()
x = datasets['data']
y = datasets.target

print(x.shape,y.shape)  #(150, 4) (150,)


# pca를 하기전에 스케일링이 필요함 보통 standardscaler를 사용
# pca = PCA(n_components=2)
# x = pca.fit_transform(x)

# print(x.shape)  #(150, 4) ->(150, 2)


scaler = StandardScaler()
x = scaler.fit_transform(x)
pca = PCA(n_components=25)
x = pca.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=888,shuffle=True)

model = RandomForestClassifier(random_state=888)

model.fit(x_train,y_train)
results = model.score(x_test,y_test)
print('model.score:', results)

evr= pca.explained_variance_ratio_  #변화율 
print(evr)
print(sum(evr))

evr_cumsum=np.cumsum(evr)   #변화율합이 나오고 나온값을 보고 변화율이 적으면서 압축이 많이 되는 지점을 찾을수있음
print(evr_cumsum)

import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()

# (150, 4) (150,)
# model.score: 1.0

# pca 적용
# (150, 2) (150,)
# model.score: 0.8666666666666667