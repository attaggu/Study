from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import time
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

(x_train,y_train),(x_test,y_test) = mnist.load_data()  #pca에 y는 필요없음

start_time=time.time()
x = np.concatenate([x_train, x_test], axis=0)
y= np.concatenate([y_train,y_test],axis=0)
print(x.shape)  #(70000, 28, 28)

x =x.reshape(-1,28*28)
# y =y.reshape(-1,1)

components=[154,331,486,713,-1]
rs = 1212
models =[
    # RandomForestClassifier(random_state=rs),
    # GradientBoostingClassifier(random_state=rs),
    XGBClassifier(random_state=rs)
    ]

for num in components:
    if num ==-1:
        pca = PCA()
    else:
        pca=PCA(n_components=num)
    xpca=pca.fit_transform(x)
    
    x_train,x_test,y_train,y_test = train_test_split(xpca,y,train_size=0.8)
    for model in models:
        model.fit(x_train,y_train)
        result=model.score(x_test,y_test)
        print("model.score:",result)
        y_predict=model.predict(x_test)
        acc=accuracy_score(y_test,y_predict)
        print(model, "acc", acc)
        print(type(model).__name__, ":",model.feature_importances_) 
    
    print(f"Number of components: {num}")
    print("loss:" , result)
    
end_time=time.time()
print("time:",round(end_time-start_time,2), "초")
