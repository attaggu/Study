# sclaer , paca 후 split

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
print(sk.__version__)   #1.1.3

datasets = load_wine()
x = datasets['data']
y = datasets.target

print(x.shape,y.shape)  #(569, 30) (569,)


# pca를 하기전에 스케일링이 필요함 보통 standardscaler를 사용
# pca = PCA(n_components=2)
# x = pca.fit_transform(x)

scaler = StandardScaler()
xs = scaler.fit_transform(x)


max_len_components = x.shape[1]
components = range(1, max_len_components + 1)
# components = list(range(1, max_len_components + 1))

for n_componets in components:
    pca = PCA(n_components= n_componets)
    x=pca.fit_transform(xs)
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=888,shuffle=True,stratify=y)
    model = RandomForestClassifier(random_state=888)
    
    model.fit(x_train,y_train)
    results = model.score(x_test,y_test)
    # print(f'n_components={components}, model.score: {results}')
    print(x_train.shape , results)

evr= pca.explained_variance_ratio_  #변화율 
print(evr)
print(sum(evr))

evr_cumsum=np.cumsum(evr)   #변화율합이 나오고 나온값을 보고 변화율이 적으면서 압축이 많이 되는 지점을 찾을수있음
print(evr_cumsum)  

#(142, 9) 1.0  /  0.94239698