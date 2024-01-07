from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import time
#1data

datasets = load_diabetes()
x = datasets.data
y = datasets.target 
print(x)
print(y)
print(x.shape,y.shape)  #(442,10) (442,)
print(datasets.feature_names)   #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)   #설명

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 random_state=563)
model = Sequential()
model.add(Dense(10,input_dim=10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train,y_train, epochs=4000, batch_size=100)
end_time = time.time()
loss=model.evaluate(x_test,y_test)

y_predict=model.predict(x_test)
result=model.predict(x)
r2 = r2_score(y_test,y_predict)
print("loss:",loss)
print("R2_score:",r2)
print("time:",round(end_time-start_time,2),"s")
