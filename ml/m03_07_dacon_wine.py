from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier





path = "c://_data//dacon//wine//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv",index_col=0)
# print(test_csv)
submission_csv=pd.read_csv(path + "sample_submission.csv")

train_csv['type']=train_csv['type'].map({'white':1,'red':0}).astype(int)
test_csv['type']=test_csv['type'].map({'white':1,'red':0}).astype(int)

x=train_csv.drop(['quality'],axis=1)
y=train_csv['quality']



x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,
                                               random_state=6112, stratify= y)
# model=LinearSVC()
# model = Perceptron()
# model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()
#Best RandomForestClassifier


hist=model.fit(x_train,y_train)

result=model.score(x_test,y_test)
print("model.score:",result)

y_predict=model.predict(x_test)
y_submit=model.predict(test_csv)


submission_csv['quality']= y_submit
submission_csv.to_csv(path + "sample_submission_2.csv",index=False)
y_submit=y_submit+3

def ACC(a,b):
    return accuracy_score(a,b)
acc=ACC(y_test,y_predict)
print("score:",acc)



# plt.figure(figsize=(10,10))
# plt.plot(hist.history['acc'],c='red',label='acc',marker='.')
# plt.plot(hist.history['val_acc'],c='blue',label='val_acc',marker='.')
# plt.legend(loc='upper right')
# plt.title('wine quality')
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.grid()
# plt.show()