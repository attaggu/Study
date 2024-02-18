import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,BatchNormalization, AveragePooling1D, Flatten, Conv2D, LSTM, Bidirectional,Conv1D,MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier

path= "c:/_data/kaggle/새 폴더/"
train=pd.read_csv(path+"train.csv",index_col=0)
test=pd.read_csv(path+"test.csv",index_col=0)
sample=pd.read_csv(path+"sample_submission.csv")
x= train.drop(['NObeyesdad'],axis=1)
y= train['NObeyesdad']
# print(train.shape,test.shape)   #(20758, 17) (13840, 16)  
# print(x.shape,y.shape)  #(20758, 16) (20758,)

lb = LabelEncoder()


columns_to_encode = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']


for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])


for column in columns_to_encode:
    lb.fit(test[column])
    test[column] = lb.transform(test[column])

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print("1사분위 :", quartile_1)
    print("q2", q2)
    print("3사분위 :", quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr:", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)    
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound) | (data_out < lower_bound))
outliers_loc = outliers(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.78,random_state=66101,stratify=y)

# print(x_train.shape,y_train.shape)  #(18682, 16) (18682,)
# print(x_test.shape,y_test.shape)    #(2076, 16) (2076,)

import random

r = random.randint(1, 100)
random_state = r
lgbm_params = {"objective": "multiclass",
               "metric": "multi_logloss",
               "verbosity": -1,
               "boosting_type": "gbdt",
               "random_state": random_state,
               "num_class": 7,
               "learning_rate" :  0.01386432121252535,
               'n_estimators': 77,         #에포
               'feature_pre_filter': False,
               'lambda_l1': 1.2149501037669967e-07,
               'lambda_l2': 0.9230890143196759,
               'num_leaves': 31,
               'feature_fraction': 0.5,
               'bagging_fraction': 0.5523862448863431,
               'bagging_freq': 3,
               'min_child_samples': 20}

model = LGBMClassifier(**lgbm_params)


model.fit(x_train, y_train)
model.booster_.save_model("c:\_data\_save\비만5.h5")

y_pred = model.predict(x_test)
y_submit = model.predict(test)
sample['NObeyesdad']=y_submit

sample.to_csv(path + "tae5.csv", index=False)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("r",r)

# Accuracy: 0.895774031092621
# r 81


# q2 1.394539
# iqr: 2.060018
# Accuracy: 0.895117144733961
# r 74