from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split,cross_val_score,KFold,StratifiedKFold
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.svm import LinearSVC, SVC

# 1. Data
x,y = load_breast_cancer(return_X_y=True)
# datasets = load_breast_cancer()
# x=datasets.data
# y=datasets.target
n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=28)
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=28)

model=SVC()

scores = cross_val_score(model, x, y ,cv=kfold)

print("acc:",scores, "\ncv-acc:", round(np.mean(scores),4))

