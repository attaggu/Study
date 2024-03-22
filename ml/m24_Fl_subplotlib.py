import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class CustomXGBClassifier(XGBClassifier):
    def __str__(self):
        return 'XGBClassifier()'

# 1. Data


datasets=load_iris()
x=datasets.data
y=datasets['target']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=28,stratify=y)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# 2. Model

rs = 1212
model1 =DecisionTreeClassifier(random_state=rs)
model2 = RandomForestClassifier(random_state=rs)
model3 = GradientBoostingClassifier(random_state=rs)
model4 = CustomXGBClassifier(random_state=rs)
models =[model1,model2,model3,model4]

for model in models:
    model.fit(x_train,y_train)
    result= model.score(x_test,y_test)
    print("model.score:",result)
    y_predict=model.predict(x_test)
    acc=accuracy_score(y_test,y_predict)
    # print(model, "acc", acc)
    # print(model.feature_importances_) 
    # print(type(model).__name__, ":",model.feature_importances_)   
 


def plot_feature_importances_datasets(model,color):
    n_features = len(model.feature_importances_)
    plt.barh(np.arange(n_features),model.feature_importances_,align='center', color=color)
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)
    plt.title(model)

colors= ['r','g','y','b']
i=1
for model,color in zip(models, colors) :
    plt.subplot(2,2,i)
    plot_feature_importances_datasets(model,color)
    i+=1

plt.show()

# plt.subplot(2,2,1)
# plot_feature_importances_datasets(model1)
# plt.subplot(2,2,2)
# plot_feature_importances_datasets(model2)
# plt.subplot(2,2,3)
# plot_feature_importances_datasets(model3)
# plt.subplot(2,2,4)
# plot_feature_importances_datasets(model4)

# plt.show()




# class CustomXGBClassifier(XGBClassifier):
#     def __str__(self):
#         return 'XGBClassifier()'

# print(model, ":", model.feature_importances_)   




# print(model, "acc", acc)

# print(model.feature_importances_)   
#[0.04519231 0.         0.54879265 0.40601504] - 각각 feature 점수
# print(model, ":",model.feature_importances_)   
#DecisionTreeClassifier(random_state=1212) : [0.04519231 0.         0.54879265 0.40601504]
#RandomForestClassifier(random_state=1212) : [0.09680396 0.02696853 0.39722367 0.47900384]
#GradientBoostingClassifier(random_state=1212) : [0.01522103 0.0134919  0.27358378 0.6977033 ]
# XGBClassifier(random_state=1212) : [0.01192079 0.02112738 0.51003134 0.45692044]