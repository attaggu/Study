
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM,Conv1D,Flatten
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from keras.callbacks import EarlyStopping


datasets = load_boston()
x = datasets.data   #.data= x
y = datasets.target #.target= y
import warnings
warnings.filterwarnings('ignore')   #ignore = warnings 안보게할때
print(x.shape)  #(506,13)
print(y.shape)  #(506,)
print(datasets.feature_names)
# ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.9,
                                                 random_state=37)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

print(np.min(x_train))  #0
print(np.max(x_train))  #1
print(np.min(x_test))
print(np.max(x_test))


model = Sequential()
# model.add(LSTM(16, input_shape=(13,1)))
model.add(Conv1D(16,kernel_size=2,input_shape=(13,1)))
model.add(Flatten())  
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# es=EarlyStopping(monitor='val_acc',mode='min',patience=400,verbose=1,
                #  restore_best_weights=True)
model.fit(x_train,y_train, epochs=1000, batch_size=50,
          validation_split=0.2,
        #   callbacks=[es]
          )

loss = model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
result=model.predict(x)
r2 = r2_score(y_test,y_predict)
print("loss:",loss)
print("R2_score:",r2)

# loss: [4.944716930389404, 0.0]
# R2_score: 0.4353930567717563