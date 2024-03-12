from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer

import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

datasets = load_breast_cancer()
x = datasets.data   #.data= x
y = datasets.target #.target= y
import warnings
warnings.filterwarnings('ignore')   #ignore = warnings 안보게할때
print(x.shape)  #(569, 30)
print(y.shape)  #(569,)



x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.9,
                                                 random_state=37,stratify=y)

scaler = MinMaxScaler() 
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


model = Sequential()
model.add(Dense(16, input_dim=30))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1,activation='sigmoid'))

from keras.optimizers import Adam
lr = 0.01


hist=model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr))
hist=model.fit(x_train,y_train, epochs=10, batch_size=50,
          validation_split=0.2,
         
          )


loss = model.evaluate(x_test,y_test,verbose=0)
y_predict=model.predict(x_test,verbose=0)
r2 = r2_score(y_predict,y_test)
print("lr : {0}, loss : {1}".format(lr,loss))
print("lr : {0}, loss : {1}".format(lr, r2))


# lr : 0.01, loss : 0.09190893918275833
# lr : 0.01, loss : 0.8526332410199446