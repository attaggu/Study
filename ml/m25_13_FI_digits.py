import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
class CustomXGBClassifier(XGBClassifier):
    def __str__(self):
        return 'XGBClassifier()'
# aaa = CustomXGBClassifier()

# 1. Data
# x, y=load_iris(return_X_y=True)
datasets=load_digits()
x=datasets.data
y=datasets.target


# df=pd.DataFrame(x,columns=datasets.feature_names)

# x=df.drop(df.columns[0],axis=1)

# pd.DataFrame
# 컬럼명 : datasets.feature_names
print(x.shape,y.shape)
 
feature_importances = np.array([0.00000000e+00, 2.04868169e-03, 1.96248010e-02, 1.16165733e-02, 9.17970161e-03, 2.17732329e-02, 9.26480090e-03, 8.99487976e-04, 6.68528363e-05, 1.03349562e-02, 2.89055336e-02, 7.14101247e-03, 1.74140086e-02, 2.97615945e-02, 5.90585338e-03, 4.82463765e-04, 1.44885330e-05, 6.91231067e-03, 1.85601801e-02, 2.87978848e-02, 3.08498491e-02, 5.63135879e-02, 7.89223691e-03, 7.18852623e-04, 7.72368990e-05, 1.41257542e-02, 4.44042166e-02, 2.52641533e-02, 3.69064261e-02, 2.27826899e-02, 2.97133520e-02, 7.26614329e-05, 0.00000000e+00, 3.13667497e-02, 2.46402866e-02, 1.74065210e-02, 3.42372609e-02, 1.82547728e-02, 2.30402338e-02, 0.00000000e+00, 3.89623827e-05, 1.03954024e-02, 3.24027587e-02, 4.50911234e-02, 1.94760309e-02, 1.75015798e-02, 1.86267305e-02, 1.10332774e-04, 0.00000000e+00, 2.20255434e-03, 1.67518387e-02, 1.83659002e-02, 1.46395343e-02, 2.35364627e-02, 2.81350642e-02, 1.65576813e-03, 0.00000000e+00, 1.53020520e-03, 2.08037161e-02, 1.14096053e-02, 2.07444265e-02, 2.86784189e-02, 1.77509536e-02, 3.38137054e-03])



sorted_indices = np.argsort(feature_importances)
num_cols_to_keep = int(len(feature_importances) * 0.8)
indices_to_keep = sorted_indices[-num_cols_to_keep:]
x = x[:, indices_to_keep]

print(y)


print(x.shape,y.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=28,stratify=y)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# 2. Model

rs = 1212
models =[DecisionTreeClassifier(random_state=rs),
    RandomForestClassifier(random_state=rs),
    GradientBoostingClassifier(random_state=rs),
    # XGBClassifier(random_state=rs)
    CustomXGBClassifier(random_state=rs)
    ]

for model in models:
    model.fit(x_train,y_train)
    result= model.score(x_test,y_test)
    print("model.score:",result)
    y_predict=model.predict(x_test)
    acc=accuracy_score(y_test,y_predict)
    # print(model, "acc", acc)
    # print(model.feature_importances_) 
    # print(type(model).__name__, ":",model.feature_importances_)   
    print(model, ":", model.feature_importances_)   

# class CustomXGBClassifier(XGBClassifier):
#     def __str__(self):
#         return 'XGBClassifier()'

# print(model, ":", model.feature_importances_)   

