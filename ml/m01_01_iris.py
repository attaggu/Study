import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.svm import LinearSVC


# 1. Data
datasets=load_iris()
x=datasets.data
y=datasets.target




x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 test_size=0.2,
                                                 random_state=2727,
                                                 stratify=y,    #y값 숫자를 비율대로 고정(분류에서만 사용)
                                                 )

# 2. Model
# model=Sequential()
# model.add(Dense(1,input_dim=4)) 
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))   
model = LinearSVC(C=100)
#C가 크면 training 포인트를 정확히 구분(굴곡짐), C가 작으면 직선에 가깝다.

# 3. Compile, Fit
# model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',
#               metrics=['acc'])
model.fit(x_train,y_train)

# 4. Evaluate, Predict
# result=model.evaluate(x_test,y_test)
result= model.score(x_test,y_test)
print("model.score:",result) #model.score 이기 때문

y_predict=model.predict(x_test)
print(y_predict)

acc=accuracy_score(y_test,y_predict)
print("acc", acc)

'''
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict,axis=1)
'''
