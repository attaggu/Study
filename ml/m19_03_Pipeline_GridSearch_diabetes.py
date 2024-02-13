from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,HalvingGridSearchCV,RandomizedSearchCV
datasets=load_diabetes()
x=datasets.data
y=datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=123)

parameters = [
    {"RF__n_estimators": [100, 200], "RF__max_depth": [6, 10, 12], "RF__min_samples_leaf": [3, 10]},    #12
    {"RF__max_depth": [6, 8, 10, 12], "RF__min_samples_leaf": [3, 5, 7, 10]},   #16
    {"RF__min_samples_leaf": [3, 5, 7, 10], "RF__min_samples_split": [2, 3, 5, 10]},    #16
    {"RF__min_samples_split": [2, 3, 5, 10]},   #4
]
pipe = Pipeline([('MM',MinMaxScaler()),
                  ('RF',RandomForestRegressor())])
# model=GridSearchCV(pipe, parameters,cv=5,verbose=1,n_jobs=-1)
# model=RandomizedSearchCV(pipe, parameters,cv=5,verbose=1,n_jobs=-1)
model=HalvingGridSearchCV(pipe, parameters,cv=5,verbose=1,n_jobs=-1)

hist=model.fit(x_train,y_train)
loss=model.score(x_test,y_test)
y_predict=model.predict(x_test)
result=model.predict(x)
r2=r2_score(y_test,y_predict)
print("model.score:",loss)
print("r2:",r2)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE:",rmse)


# plt.rcParams['font.family']='Malgun Gothic'
# plt.rcParams['axes.unicode_minus']=False

# plt.figure(figsize=(10,10))
# plt.legend(loc='upper right')
# plt.title('당뇨병')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

# loss: 2306.89501953125
# r2: 0.6540809404569466
# RMSE: 48.0301477131255

