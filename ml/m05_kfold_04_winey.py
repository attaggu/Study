from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
x,y = load_wine(return_X_y=True)

splits = 5
fold = KFold(n_splits=splits, shuffle=True, random_state=1219)
# fold = StratifiedKFold(n_splits=splits,shuffle=True,random_state=1219)
model = SVC()

scores =cross_val_score(model,x,y,cv=fold)

print("acc :",scores , "\ncv-acc :", round(np.mean(scores),4))
