from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
#boston - boston의 집값을 찾는 데이터(방의 갯수, 방의 크기 ...)
    # pip uninstall scikit-learn
    # pip uninstall scikit-learn-intelex
    # pip uninstall scikit-image
    # pip install scikit-learn==1.1.3
datasets = load_boston()
print(datasets)
x = datasets.data   #.data= x
y = datasets.target #.target= y
import warnings
warnings.filterwarnings('ignore')   #ignore = warnings 안보게할때
print(x.shape)  #(506,13)
print(y.shape)  #(506,)
print(datasets.feature_names)
# ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
print(datasets.DESCR)   #DESCR = 데이터셋에대한 설명
#train_size 0.7이상, 0.9이하
#R2 0.62 이상

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.85,
                                                 random_state=37)
model = Sequential()
# model.add(Dense(16, input_dim=13))
model.add(Dense(10, input_shape=(13,))) #컬럼을 벡터형태로 변경 -> 13 에서 (13,)
#앞에 단위를 지움 : (10,13,15)->(13,15) / (10,14,12,16)->(14,12,16)
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train, epochs=1000, batch_size=50)
loss = model.evaluate(x_test,y_test)

y_predict=model.predict(x_test)
result=model.predict(x)
r2 = r2_score(y_test,y_predict)
print("loss:",loss)
print("R2_score:",r2)
# epochs=2000, batch_size=50, random_state=1234
# loss: 21.963037490844727
# R2_score: 0.7704061164119362

loss: 51.2153205871582
R2_score: 0.6033154039069166