from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

datasets = load_boston()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test  = train_test_split(x,y,
                                                   train_size=0.8,
                                                   random_state=1)
model=Sequential()
model.add(Dense(10,input_dim=13))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam',
             metrics=['mae'])
from keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss',
                 mode='min',
                 patience=10,
                 verbose=1,
                 restore_best_weights=True)

hist=model.fit(x_train,y_train,epochs=1000,
               batch_size=10,callbacks=[es],
               validation_split=0.2)
loss = model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)

print("loss:",loss)
print("r2:",r2)

def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))
rmse = RMSE(y_test,y_predict)

print("rmse:",rmse)