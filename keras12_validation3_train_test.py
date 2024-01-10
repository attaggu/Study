import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score,mean_absolute_error
x = np.array(range(1,17))
y = np.array(range(1,17))

# x_train=x[:7]
# x_val=x[7:13]
# x_test=x[-3:]

# y_train=y[:7]
# y_val=y[7:13]
# y_test=y[-3:]

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=False,train_size=0.625)
x_val,x_test,y_val,y_test = train_test_split(x_test,y_test,shuffle=False,test_size=0.5)
print(x_train)
print(x_val)
print(x_test)


model=Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=10,validation_data=(x_val,y_val))
                                                        
loss = model.evaluate(x_test,y_test)
result=model.predict([14,15,16])
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print("loss:",loss)
print("??:",result)
print("R2_score:",r2)