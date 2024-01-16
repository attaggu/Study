from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
# from sklearn.metrics import 
import numpy as np
import pandas as pd

x=np.array([1,2,3])
print(x)
y=np.array([1,2,3])
print(y)

model = Sequential()
model.add(Dense(5, input_shape=(1,)))
model.add(Dense(4,))
model.add(Dense(2,))
model.add(Dense(1,))
#y=w1x+w2x+w3x+b ->b도 포함돼서 계산
model.summary()
