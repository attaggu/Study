# sclaer , paca 후 split

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
print(sk.__version__)   #1.1.3
import pandas as pd
path = "c://_data//dacon//wine//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv",index_col=0)
# print(test_csv)
submission_csv=pd.read_csv(path + "sample_submission.csv")

train_csv['type']=train_csv['type'].map({'white':1,'red':0}).astype(int)
test_csv['type']=test_csv['type'].map({'white':1,'red':0}).astype(int)

x=train_csv.drop(['quality'],axis=1)
y=train_csv['quality']
print(x.shape,y.shape) 


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

#(4397, 8) 0.6681818181818182 / 0.93706722