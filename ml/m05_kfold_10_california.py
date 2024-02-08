from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.svm import LinearSVR,SVR
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

x,y=fetch_california_housing(return_X_y=True)


splits=5
# fold = KFold(n_splits=splits,shuffle=True,random_state=119)
fold = KFold(n_splits=splits,shuffle=True,random_state=119)

model = RandomForestRegressor()

scores=cross_val_score(model,x,y,cv=fold)

print("acc :",scores , "\ncv-acc :", round(np.mean(scores),4))