from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
x=np.array(range(1,17))
y=np.array(range(1,17))

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.85,shuffle=True)
# x_val,x_test,y_val,y_test=train_test_split()
print(x_train,y_train)
print(x_test,y_test)

model = Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))



model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=1,
          validation_split=0.4,
          verbose=1)
#40프로를 validataion으로 쓴다

loss = model.evaluate(x_test,y_test)
result=model.predict(y_test)
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print("loss:",loss)
print("??:",result)
print("R2_score:",r2)