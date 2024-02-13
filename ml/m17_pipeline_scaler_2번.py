#train,test 비율이 약간 애매해서 잘사용은 안함.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# 1. Data
# datasets=load_iris()
# x=datasets.data
# y=datasets.target
x, y=load_iris(return_X_y=True)
print(x.shape,y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=28,stratify=y)

print(np.min(x_train),np.max(x_train))
print(np.min(x_test),np.max(x_test))

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)


# 2. Model
# model = RandomForestClassifier()
model = make_pipeline(MinMaxScaler(),
                      StandardScaler(),
                      RobustScaler(),
                      PCA(),    #차원축소 - 데이터 손실 
                      RandomForestClassifier())

# 3. Compile, Fit
model.fit(x_train,y_train)

# 4. Evaluate, Predict
result= model.score(x_test,y_test)
print("model.score:",result)

y_predict=model.predict(x_test)
print(y_predict)

acc=accuracy_score(y_test,y_predict)
print("acc", acc)

# y_test = np.argmax(y_test, axis=1)
# y_predict = np.argmax(y_predict,axis=1)
