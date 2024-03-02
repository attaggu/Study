import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
plt.rcParams['font.family']='Malgun Gothic' #한글
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# 1. 데이터
path = "c://_data//dacon//wine//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv",index_col=0)

submission_csv=pd.read_csv(path + "sample_submission.csv")

train_csv['type']=train_csv['type'].map({'white':1,'red':0}).astype(int)
test_csv['type']=test_csv['type'].map({'white':1,'red':0}).astype(int)


x=train_csv.drop(['quality'],axis=1)
y=train_csv['quality']-3
pf = PolynomialFeatures(degree=2, include_bias=False)
x_poly = pf.fit_transform(x)

print(x_poly.shape) #(100, 2)

x_train, x_test, y_train, y_test = train_test_split(x_poly, y, random_state=777, train_size=0.8,stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. Model
model = RandomForestClassifier()

# 3. Compile Fit
model.fit(x_train,y_train)
# model2.fit(x_poly,y)


y_predict = model.predict(x_test)
# print('model.score :',model.score(x_test,y_test))
print('acc :', accuracy_score(y_test,y_predict))



'''
plt.scatter(x,y,color='blue', label='원데이터')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression Example')

x_plot = np.linspace(-1,1,100).reshape(-1,1)
x_plot_poly = pf.transform(x_plot)
y_plot = model.predict(x_plot)
y_plot2 = model2.predict(x_plot_poly)
plt.plot(x_plot,y_plot,color='red', label='Polynomial Regression')
plt.plot(x_plot,y_plot2,color='green',label='Polynomial Regression2')
plt.legend()
plt.show()
'''
# acc : 0.6745454545454546