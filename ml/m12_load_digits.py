from sklearn.model_selection import RandomizedSearchCV,train_test_split,GridSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold,cross_val_score,cross_val_predict
from sklearn.metrics import accuracy_score
import time
import pandas as pd
import numpy as np
# datasets=load_digits()
# x=datasets.data
# y=datasets.target
x, y = load_digits(return_X_y=True)
print(x)
print(y)
print(x.shape,y.shape)  #(1797, 64) (1797,)

print(pd.value_counts(y,sort=False))    #sort = False -> 숫자순서대로 
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180

# parameters = [
#     {"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
#     {"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
#     {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
#     {"min_samples_split": [2, 3, 5, 10]},
#     {"n_jobs": [-1, 2, 4], "min_samples_split": [2, 3, 5, 10]}
# ]
parameters = [
    {"n_jobs": [-1],"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"n_jobs": [-1],"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"n_jobs": [-1],"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1], "min_samples_split": [2, 3, 5, 10]}
]
x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,random_state=12,train_size=0.8)
splits=8
fold = KFold(n_splits=splits,shuffle=True,random_state=28)
model = RandomizedSearchCV(RandomForestClassifier(),parameters,cv=fold,verbose=1,
                           refit=True, n_jobs=-1)
start_time=time.time()
model.fit(x_train,y_train)
end_time=time.time()
y_predict=model.predict(x_test)
print("acc.score:", accuracy_score(y_test,y_predict))
y_pred_best=model.best_estimator_.predict(x_test)

print("best_acc.score:",accuracy_score(y_test,y_pred_best))
print("time:",round(end_time-start_time,2),"s")
print("model.score:",model.score(x_test,y_test))

# random--
# acc.score: 0.9805555555555555
# best_acc.score: 0.9805555555555555
# time: 1.87 s
# model.score: 0.9805555555555555
# grid--
# acc.score: 0.9777777777777777
# best_acc.score: 0.9777777777777777
# time: 3.23 s
# model.score: 0.9777777777777777