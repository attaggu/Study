import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from imblearn.over_sampling import SMOTE

path = "c://_data//dacon//wine//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv",index_col=0)

submission_csv=pd.read_csv(path + "sample_submission.csv")

train_csv['type']=train_csv['type'].map({'white':1,'red':0}).astype(int)
test_csv['type']=test_csv['type'].map({'white':1,'red':0}).astype(int)


x=train_csv.drop(['quality'],axis=1)
y=train_csv['quality']-3
print(train_csv['quality'].value_counts().sort_index())
print(x.shape,y.shape)  #(5497, 12) (5497,)

x,_,y,_ = train_test_split(x,y,train_size=0.9,random_state=44,stratify=y)


print(x.shape,y.shape)  #(4947, 12) (4947,)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=44,stratify=y)


print(np.unique(y,return_counts=True))
print(pd.value_counts(y))


# print(np.unique(y_train,return_counts=True))


smote=SMOTE(random_state=44,k_neighbors=3)
x_train,y_train=smote.fit_resample(x_train,y_train)
print(np.unique(y_train,return_counts=True))


'''
print(y)
print("=============================================")
# x=x[30:]    #앞에서부터 30개 삭제
# y=y[30:]
# print(y)

# print(np.unique(y,return_counts=True))
x=x[:-35]   #뒤에서부터 30개 삭제
y=y[:-35]
print(y)
print(np.unique(y,return_counts=True))
#(array([0, 1, 2]), array([59, 71, 18], dtype=int64))

print(np.unique(y,return_counts=True))



#model
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,
                                               shuffle=True,
                                               random_state=14,
stratify=y)

print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)




'''