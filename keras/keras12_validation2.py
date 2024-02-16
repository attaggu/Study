import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error

x = np.array(range(1,17))
y = np.array(range(1,17))

x_train=x[:7]
x_val=x[7:13]
x_test=x[-3:]

y_train=y[:7]
y_val=y[7:13]
y_test=y[-3:]


model=Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=50)

model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=10,validation_data=(x_val,y_val))
                                                        
loss = model.evaluate(x_test,y_test)
result=model.predict([14,15,16])
print("loss:",loss)
print("??:",result)
# print("R2_score:",r2)