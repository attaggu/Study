from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau



datasets = load_breast_cancer()
x = datasets.data   #.data= x
y = datasets.target #.target= y


x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.9,
                                                 random_state=37,stratify=y)



# 2. model

model = Sequential()
model.add(Dense(16, input_dim=30))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1,activation='sigmoid'))


from keras.optimizers import Adam
learning_rate = 0.0001

hist=model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate))
hist=model.fit(x_train,y_train, epochs=100, batch_size=50,
          validation_split=0.2,
          )

# model=load_model('../_data/_save/MCP/keras25_MCP3_.hdf5')
#가중치와 모델 전부 저장돼어 불러와진다

loss = model.evaluate(x_test,y_test)
y_predict=np.round(model.predict(x_test))
acc = accuracy_score(y_test,y_predict)
print("lr : {0}, loss : {1}".format(learning_rate,loss))
print("lr : {0}, loss : {1}".format(learning_rate, acc))



# lr : 0.0001, loss : 0.0941818580031395
# lr : 0.0001, loss : 0.9649122807017544

# lr : 0.001, loss : 0.5321984887123108 
# lr : 0.001, loss : 0.8070175438596491

# lr : 0.01, loss : 0.12863540649414062 
# lr : 0.01, loss : 0.9473684210526315 

# lr : 0.1, loss : 0.07621367275714874  
# lr : 0.1, loss : 0.9473684210526315 

# lr : 1.0, loss : 0.33366474509239197  
# lr : 1.0, loss : 0.9298245614035088 