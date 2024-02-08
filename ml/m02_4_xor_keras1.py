import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense


x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([0, 1, 1, 0 ])
print(x_data.shape, y_data.shape)   #(4, 2) (4,)

model= Sequential()
model.add(Dense(1,input_dim=2, activation= 'sigmoid'))


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_data,y_data)

result= model.evaluate(x_data,y_data)

print("model.evaluate:", result[0])
print("acc:",result[1])


y_predict = model.predict(x_data)
y_predict = np.around(y_predict).reshape(-1,).astype(int)
acc2=accuracy_score(y_data,y_predict)
print("acc:",acc2)
print("======================")
print(y_data)
print(y_predict)