import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

# 1. Data
datasets=load_iris()
# print(datasets) # (n,4) input 4
# print(datasets.DESCR)   #4col,3class
# print(datasets.feature_names)

x=datasets.data
y=datasets.target
# print(x.shape,y.shape)  #(150, 4) (150,)

# 1. keras
from keras.utils import to_categorical
# y_ohe = to_categorical(y)
# print(y_ohe)
# print(y_ohe.shape)

# # 2. pandas
# y_ohe2 = pd.get_dummies(y)
# print(y_ohe2)
# print(y_ohe2.shape)

# 3. sklearn
from sklearn.preprocessing import OneHotEncoder
y=y.reshape(-1,1)   
ohe = OneHotEncoder()   #True 디포틀로 할시에
y_ohe3 = ohe.fit_transform(y).toarray()
ohe = OneHotEncoder(sparse=False)   #spares=True 가 디폴트
y_ohe3 = ohe.fit_transform(y)   # fit + transform 대신 쓴다 
# ==== ohe.fit(y)
# ==== y_ohe3=ohe.transform(y)

print(y_ohe3)
print(y_ohe3.shape)
