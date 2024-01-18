

#세이브 파일 만들기
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from keras.callbacks import EarlyStopping,ModelCheckpoint



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



# 2. model

model = Sequential()
model.add(Dense(16, input_dim=13))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

import datetime
date=datetime.datetime.now()
print(date) #2024-01-17 10:54:36.094603 - 
#월,일,시간,분 정도만 추출
print(type(date))   #<class 'datetime.datetime'>
date=date.strftime("%m%d-%H%M")
#%m 하면 month를 땡겨옴, %d 하면 day를 / 시간,분은 대문자
print(date) #0117_1058
print(type(date))   #<class 'str'> 문자열로 변경됨

path='../_data/_save/MCP/'  #문자를 저장
filename= '{epoch:04d}-{val_loss:.4f}.hdf5' #0~9999 : 4자리 숫자까지 에포 / 0.9999 소숫점 4자리 숫자까지 발로스
filepath= "".join([path,'k25',date,'_',filename]) # ""는 공간을 만든거고 그안에 join으로 합침 , ' _ ' 중간 공간



es=EarlyStopping(monitor='val_loss',mode='min',
                 patience=10,verbose=1,restore_best_weights=True,
                 )
mcp=ModelCheckpoint(monitor='val_loss',mode='auto',
                    verbose=1,save_best_only=True,
                    filepath=filepath   #경로저장
                    )
model.compile(loss='mae', optimizer='adam',metrics=['acc'])
hist=model.fit(x_train,y_train, epochs=1000, batch_size=50,
          validation_split=0.2,
          callbacks=[es,mcp],
          )
print(hist.history['loss'])     #loss만 따로
print(hist.history['val_loss'])     #val_loss만 따로

# model=load_model('../_data/_save/MCP/keras25_MCP3_.hdf5')
#가중치와 모델 전부 저장돼어 불러와진다

print("==================1.기본출력==================")
loss = model.evaluate(x_test,y_test,verbose=0)
y_predict=model.predict(x_test,verbose=0)
r2 = r2_score(y_predict,y_test)
print("loss:",loss)
print("R2_score:",r2)
