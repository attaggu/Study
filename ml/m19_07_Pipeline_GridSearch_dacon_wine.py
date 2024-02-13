from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,HalvingGridSearchCV,RandomizedSearchCV


path = "c://_data//dacon//wine//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv",index_col=0)
# print(test_csv)
submission_csv=pd.read_csv(path + "sample_submission.csv")

train_csv['type']=train_csv['type'].map({'white':1,'red':0}).astype(int)
test_csv['type']=test_csv['type'].map({'white':1,'red':0}).astype(int)

x=train_csv.drop(['quality'],axis=1)
y=train_csv['quality']



x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,
                                               random_state=6112, stratify= y)
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


submission_csv['quality']= y_submit
submission_csv.to_csv(path + "sample_submission_2.csv",index=False)
y_submit=y_submit+3

def ACC(a,b):
    return accuracy_score(a,b)
acc=ACC(y_test,y_predict)
print("score:",acc)



# plt.figure(figsize=(10,10))
# plt.plot(hist.history['acc'],c='red',label='acc',marker='.')
# plt.plot(hist.history['val_acc'],c='blue',label='val_acc',marker='.')
# plt.legend(loc='upper right')
# plt.title('wine quality')
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.grid()
# plt.show()