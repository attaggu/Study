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
# 1. Data
datasets=load_iris()
# print(datasets) # (n,4) input 4
# print(datasets.DESCR)   #4col,3class
# print(datasets.feature_names)

x=datasets.data
y=datasets.target
# print(x.shape,y.shape)  #(150, 4) (150,)
# print(y)
# print(np.unique(y,return_counts=True))  #(array([0, 1, 2]), array([50, 50, 50], dtype=int64))
# print(pd.value_counts(y))
y_ohe= to_categorical(y)



x_train,x_test,y_train,y_test = train_test_split(x,y_ohe,
                                                 test_size=0.2,
                                                 random_state=2727,
                                                 stratify=y,    #y값 숫자를 비율대로 고정(분류에서만 사용)
                                                 )

# 2. Model
model=Sequential()
model.add(Dense(1,input_dim=4)) 
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))   

# 3. Compile, Fit
model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics=['acc']) #accuracy=acc=정확도 - 가중치에 반영은 안됨 
                                    #mse,mae도 같이 넣을수 있음

es=EarlyStopping(monitor='val_loss',    #vall_acc도 상관 없음
                 mode='min',
                 patience=20,verbose=1,
                 restore_best_weights=True)
hist=model.fit(x_train,y_train,epochs=200,batch_size=5,
              callbacks=[es],
              validation_split=0.2)

# 4. Evaluate, Predict
result=model.evaluate(x_test,y_test)
print("loss:",result[0])
print("acc:",result[1])
# loss: 0.2109421044588089
# acc: 0.8999999761581421

y_predict=model.predict(x_test)
# print(y_predict)
print(y_predict.shape,y_test.shape) #(30, 3) (30, 3)

#OneHot 된 데이터를 원래대로 정수형으로 변경
y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict,axis=1)

print(y_test)
print(y_predict)
print(y_test.shape,y_predict.shape) #(30,) (30,)

acc=accuracy_score(y_predict,y_test)
print("accuracy_score:",acc)