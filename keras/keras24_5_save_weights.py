from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
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
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

print(np.min(x_train))  #0
print(np.max(x_train))  #1S
print(np.min(x_test))
print(np.max(x_test))



# 2. model

model = Sequential()
model.add(Dense(16, input_dim=13))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))
model.save_weights("..//_data//_save//keras24_5_save_weights1.h5")

# model=load_model(".//_save//keras24_3_save_model2.h5")

# model.summary()

# model.save("..//_data//_save//keras24_save_model.h5")
#c:Study에_data파일이 생기고 _save파일이 생김

#.는 현재파일
#..는 현재파일의 상위파일

model.compile(loss='mae', optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train, epochs=100, batch_size=50,
          validation_split=0.2,)
model.save_weights("..//_data//_save//keras24_5_save_weights2.h5")




loss = model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
result=model.predict(x)
r2 = r2_score(y_test,y_predict)
print("loss:",loss)
print("R2_score:",r2)

# loss: [4.944716930389404, 0.0]
# R2_score: 0.4353930567717563
# PS C:\Study> 
