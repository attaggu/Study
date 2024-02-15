from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import time

(x_train,y_train),(x_test,y_test) = mnist.load_data()  #pca에 y는 필요없음

start_time=time.time()
x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train,y_test],axis=0)
print(x.shape)  #(70000, 28, 28)

x = x.reshape(-1,28*28)

components=[154, 331, 486, 713, -1]
rs = 1212

models = [
    # RandomForestClassifier(random_state=rs),
    # GradientBoostingClassifier(random_state=rs),
    XGBClassifier(random_state=rs,
                  tree_method='hist',device='cuda'
                  )
]

parameters = [
    {"n_estimators":[100,200,300],"learning_rate":[0.1,0.3,0.001,0.01],"max_depth":[4,5,6]},
    {"n_estimators":[90,100,110],"learning_rate":[0.1,0.001,0.01],"max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[90,110],"learning_rate":[0.1,0.001,0.5],"max_depth":[4,5,6],"colsample_bytree":[0.1,0.001,0.5],"colsample_bylevel":[0.6,0.7,0.9]}
]

for num in components:
    if num == -1:
        pca = PCA()
    else:
        pca = PCA(n_components=num)
    xpca = pca.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(xpca, y, train_size=0.8)
    
    for model, params in zip(models, parameters):
        grid_search = RandomizedSearchCV(model, params, cv=3, n_jobs=-1,verbose=1)
        grid_search.fit(x_train, y_train)
        
        result = grid_search.score(x_test, y_test)
        print("model.score:", result)
        print("Best parameters:", grid_search.best_params_)
    
    print(f"Number of components: {num}")
    print("loss:" , result)
    
end_time=time.time()
print("time:", round(end_time-start_time, 2), "초")