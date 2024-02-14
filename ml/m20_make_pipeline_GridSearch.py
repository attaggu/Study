import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler
from sklearn.pipeline import make_pipeline,Pipeline     #Pipleline - class / make_pipeline - 함수
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,HalvingGridSearchCV,RandomizedSearchCV
# 1. Data
x, y=load_iris(return_X_y=True)
print(x.shape,y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=28,stratify=y)

print(np.min(x_train),np.max(x_train))
print(np.min(x_test),np.max(x_test))

parameters = [
    {"randomforestclassifier__n_estimators": [100, 200], "randomforestclassifier__max_depth": [6, 10, 12], "randomforestclassifier__0min_samples_leaf": [3, 10]},    #12
    {"randomforestclassifier__max_depth": [6, 8, 10, 12], "randomforestclassifier__min_samples_leaf": [3, 5, 7, 10]},   #16
    {"randomforestclassifier__min_samples_leaf": [3, 5, 7, 10], "randomforestclassifier__min_samples_split": [2, 3, 5, 10]},    #16
    {"randomforestclassifier__min_samples_split": [2, 3, 5, 10]},   #4
]
pipe = make_pipeline(MinMaxScaler(),RandomForestClassifier())

#====Pipeline 과 make_pipeline 차이====

# parameters = [
#     {"RF__n_estimators": [100, 200], "RF__max_depth": [6, 10, 12], "RF__min_samples_leaf": [3, 10]},    #12
#     {"RF__max_depth": [6, 8, 10, 12], "RF__min_samples_leaf": [3, 5, 7, 10]},   #16
#     {"RF__min_samples_leaf": [3, 5, 7, 10], "RF__min_samples_split": [2, 3, 5, 10]},    #16
#     {"RF__min_samples_split": [2, 3, 5, 10]},   #4
# ]
# pipe = Pipeline([('MM',MinMaxScaler()),
#                   ('RF',RandomForestClassifier())])


# model=GridSearchCV(pipe, parameters,cv=5,verbose=1,n_jobs=-1)
# model=RandomizedSearchCV(pipe, parameters,cv=5,verbose=1,n_jobs=-1)
model=HalvingGridSearchCV(pipe, parameters,cv=5,verbose=1,n_jobs=-1)

# 3. Compile, Fit
model.fit(x_train,y_train)

# 4. Evaluate, Predict
result= model.score(x_test,y_test)
print("model.score:",result)

y_predict=model.predict(x_test)
print(y_predict)

acc=accuracy_score(y_test,y_predict)
print("acc", acc)

# y_test = np.argmax(y_test, axis=1)
# y_predict = np.argmax(y_predict,axis=1)
