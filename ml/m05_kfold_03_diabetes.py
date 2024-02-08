from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.svm import LinearSVR,SVR,SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
x,y = load_diabetes(return_X_y=True)

splits = 5
fold = KFold(n_splits=splits, shuffle=True, random_state=28)

model = SVR()

scores = cross_val_score(model,x,y,cv=fold)

print("acc :",scores , "\ncv-acc :", round(np.mean(scores),4))