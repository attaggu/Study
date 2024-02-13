from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,HalvingGridSearchCV,RandomizedSearchCV

path = "c://_data//dacon//iris//"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
# print(train_csv)
test_csv = pd.read_csv(path + "test.csv",index_col=0)
# print(test_csv)
submission_csv=pd.read_csv(path + "sample_submission.csv")

x=train_csv.drop(['species'],axis=1)
y=train_csv['species']



x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,
                                               random_state=112)

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

hist=model.fit(x_train,y_train)
result=model.score(x_test,y_test)
print("model.score:",result)

y_predict=model.predict(x_test)
y_submit=model.predict(test_csv)
submission_csv['species']=y_submit

submission_csv.to_csv(path + "sample_submission_1.csv",index=False)
# y_test = np.argmax(y_test, axis=1)
# y_predict = np.argmax(y_predict,axis=1)

# print(y_test)
# print(y_predict)
# print(y_test.shape,y_predict.shape)
def ACC(a,b):
    return accuracy_score(a,b)
acc=ACC(y_test,y_predict)
print("score:",acc)

