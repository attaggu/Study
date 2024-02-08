from sklearn.model_selection import train_test_split,KFold,cross_val_score,cross_val_predict,StratifiedKFold
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVR,SVR
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
# x, y = load_boston(return_X_y=True)
datasets = load_boston()
x = datasets.data  #.data= x
y = datasets.target #.target= y

x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,random_state=121)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

splits= 4
fold = KFold(n_splits=splits,shuffle=True,random_state=2929)
model=SVR()
scores=cross_val_score(model,x_train,y_train,cv=fold)
print("acc :",scores , "\ncv-acc :", round(np.mean(scores),4))
y_predict=cross_val_predict(model,x_test,y_test,cv=fold)

# acc=accuracy_score(y_test,y_predict)
# print("acc:",acc)

