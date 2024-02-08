from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split,cross_val_score,KFold,StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier





path = "c://_data//dacon//wine//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv",index_col=0)
# print(test_csv)
submission_csv=pd.read_csv(path + "sample_submission.csv")

train_csv['type']=train_csv['type'].map({'white':1,'red':0}).astype(int)
test_csv['type']=test_csv['type'].map({'white':1,'red':0}).astype(int)

x=train_csv.drop(['quality'],axis=1)
y=train_csv['quality']

splits = 4
# fold= KFold(n_splits=splits,shuffle=True,random_state=29)
fold = StratifiedKFold(n_splits=splits,shuffle=True,random_state=29)
model = SVC()

scores=cross_val_score(model,x,y,cv=fold)
print("acc :",scores , "\ncv-acc :", round(np.mean(scores),4))
