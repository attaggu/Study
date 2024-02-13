from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,HalvingGridSearchCV,RandomizedSearchCV

# 1. Data
datasets = load_breast_cancer()

x=datasets.data
y=datasets.target


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,
                                                 random_state=28)
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

model.fit(x_train,y_train)

# 4. Evaluate, Predict
loss=model.score(x_test,y_test)

# y_predict=np.around(model.predict(x_test))
y_predict=model.predict(x_test)

r2=r2_score(y_test,y_predict)

def ACC(a,b):
    return accuracy_score(a,b)
acc=ACC(y_test,y_predict)


print("ACC:",acc)
print("model.score:",loss)
print("r2:",r2)
print("acc:",acc)
