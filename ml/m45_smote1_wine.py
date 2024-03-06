import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense

datasets=load_wine()

x=datasets.data
y=datasets['target']    #둘다 가능

print(x.shape,y.shape)  #(178, 13) (178,)

print(np.unique(y,return_counts=True))
print(pd.value_counts(y))

print(y)
print("=============================================")
# x=x[30:]    #앞에서부터 30개 삭제
# y=y[30:]
# print(y)

# print(np.unique(y,return_counts=True))
x=x[:-35]   #뒤에서부터 30개 삭제
y=y[:-35]
print(y)
print(np.unique(y,return_counts=True))
#(array([0, 1, 2]), array([59, 71, 18], dtype=int64))

print(np.unique(y,return_counts=True))



#model
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,
                                               shuffle=True,
                                               random_state=14,
stratify=y)





# model=Sequential()
# model.add(Dense(10,input_shape=(13,)))
# model.add(Dense(3,activation='softmax'))
# #compile/fit
# model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
# #========================================
# #sparse_categorical_crossentropy 를 사용하면 원핫을 안해도 다중분류가 가능
# #========================================
# model.fit(x_train,y_train,epochs=100,validation_split=0.2)
# #evaluate/predict
# result=model.evaluate(x_test,y_test)
# # print("loss:",result[0])
# # print("acc:",result[1])
# y_predict=model.predict(x_test)

# print(y_test)   #원핫 안돼있음
# print(y_predict)#원핫 돼있음

# y_predict=np.argmax(y_predict,axis=1)
# print(y_predict)
# acc=accuracy_score(y_test,y_predict)
# f1_score=f1_score(y_test,y_predict,average='macro')
# #f1 score 는 이진분류용이다 / macro - 다중분류 각각을 이중분류로 비교해서 n빵함
# print("acc:",acc)
# print("f1:",f1_score)

# epochs = 100
# acc: 0.8888888888888888
# f1: 0.863762855414398

print("==============SMOTE=============")
# smote 증폭 시작

from imblearn.over_sampling import SMOTE

# import sklearn as sk

# print("scikit:",sk.__version__) # 1.1.3

smote=SMOTE(random_state=123)
x_train,y_train=smote.fit_resample(x_train,y_train)

print(pd.value_counts(y_train))

# smote 증폭 끝


model=Sequential()
model.add(Dense(10,input_shape=(13,)))
model.add(Dense(3,activation='softmax'))
#compile/fit
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
#========================================
#sparse_categorical_crossentropy 를 사용하면 원핫을 안해도 다중분류가 가능
#========================================
model.fit(x_train,y_train,epochs=100,validation_split=0.2)
#evaluate/predict
result=model.evaluate(x_test,y_test)
# print("loss:",result[0])
# print("acc:",result[1])
y_predict=model.predict(x_test)

print(y_test)   #원핫 안돼있음
print(y_predict)#원핫 돼있음

y_predict=np.argmax(y_predict,axis=1)
print(y_predict)
acc=accuracy_score(y_test,y_predict)
f1_score=f1_score(y_test,y_predict,average='macro')
#f1 score 는 이진분류용이다 / macro - 다중분류 각각을 이중분류로 비교해서 n빵함
print("acc:",acc)
print("f1:",f1_score)
