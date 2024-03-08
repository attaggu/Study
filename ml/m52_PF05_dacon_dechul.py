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
path = "C:\\_data\\dacon\\dechul\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0 )
print(train_csv.shape)  
test_csv = pd.read_csv(path + "test.csv", index_col=0 )
print(test_csv.shape) 
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv.shape)  
train_csv = train_csv[train_csv['주택소유상태'] != 'ANY']
test_csv.loc[test_csv['대출목적'] == '결혼' , '대출목적'] = '기타'
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder()
train_csv['주택소유상태'] = le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = le.fit_transform(train_csv['대출목적'])
train_csv['대출기간'] = train_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
train_csv['근로기간'] = le.fit_transform(train_csv['근로기간'])

test_csv['주택소유상태'] = le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = le.fit_transform(test_csv['대출목적'])
test_csv['대출기간'] = test_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
test_csv['근로기간'] = le.fit_transform(test_csv['근로기간'])

train_csv['대출등급'] = le.fit_transform(train_csv['대출등급'])

x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']

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
# acc : 0.8395555324783218