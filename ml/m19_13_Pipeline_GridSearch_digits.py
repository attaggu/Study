from sklearn.model_selection import RandomizedSearchCV,train_test_split,GridSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold,cross_val_score,cross_val_predict
from sklearn.metrics import accuracy_score
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,HalvingGridSearchCV,RandomizedSearchCV

x, y = load_digits(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,random_state=12,train_size=0.8)
parameters = [
    {"RF__n_estimators": [100, 200], "RF__max_depth": [6, 10, 12], "RF__min_samples_leaf": [3, 10]},    #12
    {"RF__max_depth": [6, 8, 10, 12], "RF__min_samples_leaf": [3, 5, 7, 10]},   #16
    {"RF__min_samples_leaf": [3, 5, 7, 10], "RF__min_samples_split": [2, 3, 5, 10]},    #16
    {"RF__min_samples_split": [2, 3, 5, 10]},   #4
]
pipe = Pipeline([('MM',MinMaxScaler()),
                  ('RF',RandomForestClassifier())])
# model=GridSearchCV(pipe, parameters,cv=5,verbose=1,n_jobs=-1)
# model=RandomizedSearchCV(pipe, parameters,cv=5,verbose=1,n_jobs=-1)
model=HalvingGridSearchCV(pipe, parameters,cv=5,verbose=1,n_jobs=-1)



start_time=time.time()
model.fit(x_train,y_train)
end_time=time.time()
y_predict=model.predict(x_test)
print("acc.score:", accuracy_score(y_test,y_predict))

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