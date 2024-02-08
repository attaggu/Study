from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import all_estimators

datasets = load_boston()
x = datasets.data   #.data= x
y = datasets.target #.target= y
import warnings
warnings.filterwarnings('ignore')   #ignore = warnings 안보게할때

# ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.9,
                                                 random_state=37)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

algo = all_estimators(type_filter='regressor')

for name, algori in algo:
    try:
        model = algori()
        model.fit(x_train,y_train)
        acc=model.score(x_test,y_test)
        print(name,'==score:',acc)
    except:
        print(name,'==error')
        continue



'''
# model=LinearSVR()
# model=LinearRegression()
# model=KNeighborsRegressor()
# model=DecisionTreeRegressor()
model=RandomForestRegressor()
# Best RandomForestRegressor

model.fit(x_train,y_train)

loss = model.score(x_test,y_test)
y_predict=model.predict(x_test)
result=model.predict(x)
r2 = r2_score(y_test,y_predict)
print("model.score:",loss)
print("R2_score:",r2)

# loss: [4.944716930389404, 0.0]
# R2_score: 0.4353930567717563
# PS C:\Study> 
'''
