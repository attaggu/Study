from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


path = "c://_data//dacon//iris//"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
# print(train_csv)
test_csv = pd.read_csv(path + "test.csv",index_col=0)
# print(test_csv)
submission_csv=pd.read_csv(path + "sample_submission.csv")

x=train_csv.drop(['species'],axis=1)
y=train_csv['species']

splits = 4
fold = KFold(n_splits=splits,shuffle=True,random_state=21)
# fold = StratifiedKFold(n_splits=splits,shuffle=True,random_state=21)
model = SVC()
scores = cross_val_score(model,x,y,cv=fold)
print("acc :",scores , "\ncv-acc :", round(np.mean(scores),4))
