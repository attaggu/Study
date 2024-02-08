import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import all_estimators

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




x_train, x_test, y_train, y_test = train_test_split(
                                                    x,
                                                    y,             
                                                    train_size=0.78,                                                   
                                                    random_state=66101,
                                                    stratify=y,
                                                    shuffle=True,
                                                    )

from sklearn.preprocessing import StandardScaler, RobustScaler 

scaler=StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


algo=all_estimators(type_filter='classifier')
for name, algori in algo:
    try:
        model=algori()
        model.fit(x_train,y_train)
        acc=model.score(x_test,y_test)
        print(name,'==score:',acc)
    except:
        print(name,'==error')
        continue
'''
model=LinearSVC(C=300)
# model = Perceptron()
# model = LogisticRegression()
# model = KNeighborsClassifier()
model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# Best DecisionTreeClassifier
model.fit(x_train, y_train)

results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict, axis=1)  
# arg_test = np.argmax(y_test, axis=1)
y_submit = model.predict(test_csv)
# submit = np.argmax(y_submit, axis=1)
submitssion = le.inverse_transform(y_submit)
      
submission_csv['대출등급'] = submitssion
# y_predict = ohe.inverse_transform(y_predict)
# y_test = ohe.inverse_transform(y_test)
f1 = f1_score(y_test, y_predict, average='macro')
acc = accuracy_score(y_test, y_predict)
print("model.score : ", results)  
print("f1 : ", f1)  
submission_csv.to_csv(path + "submissionbb_0202_1.csv", index=False)

'''
