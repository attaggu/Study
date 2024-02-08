from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, LSTM, Flatten, Embedding, Input, Concatenate, concatenate, Reshape
from sklearn.metrics import r2_score
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, Normalizer, RobustScaler, StandardScaler, MaxAbsScaler
from keras.utils import to_categorical
le = LabelEncoder()


#1. 데이터

path = "c:\\_data\\sihum\\"

samsung = pd.read_csv("c:\\_data\\sihum\\삼성 240205.csv",  encoding='EUC-KR', index_col = 0)
amore = pd.read_csv("c:\\_data\\sihum\\아모레 240205.csv",  encoding='EUC-KR', index_col = 0)

samsung= samsung[1:1001]
amore= amore[1:1001]

samsung = samsung.drop(['전일비','외인비','신용비','등락률'], axis= 1)
amore = amore.drop(['전일비','외인비','신용비','등락률'], axis= 1)

#print(samsung.shape) #(1000, 12)
#print(amore.shape) #(1000, 12)


columns_to_convert = ['시가', '고가', '저가', '종가', 'Unnamed: 6', '거래량', '금액(백만)', '개인', '기관', '외인(수량)', '외국계', '프로그램']
#print(samsung.isna().sum())
#print(amore.isna().sum())


for column in columns_to_convert:
    samsung[column] = samsung[column].str.replace(',', '').astype(float)
    amore[column] = amore[column].str.replace(',', '').astype(float)
    

print(type(samsung)) #<class 'pandas.core.frame.DataFrame'>

#print(samsung)

s_col = samsung.columns
a_col = amore.columns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
mms1 = MinMaxScaler()
samsung = mms1.fit_transform(samsung)

mms2 = MinMaxScaler()
amore = mms2.fit_transform(amore)
#print(samsung)


samsung = pd.DataFrame(samsung,columns= s_col)
amore = pd.DataFrame(amore,columns= a_col)
#print(samsung)

########데이터 뒤집기#########

samsung = samsung[::-1]
amore = amore[::-1]




timestep = 12
#720개 훈련시켜 하루 뒤 예측
predict_step = 2

def split_xy(data, timestep, y_column, predict_step=0):
    x, y = list(), list()
    
    num = len(data) - (timestep + predict_step)
    for i in range(num):
        x.append(data[i : i+ timestep])
        y.append(data.iloc[i+timestep+predict_step][y_column])
        
        
        
    return np.array(x), np.array(y)

    
    
x1, y1 = split_xy(samsung, timestep, '시가', predict_step)
x2, y2 = split_xy(amore, timestep, '종가', predict_step)

# print(x1.shape) #(993, 5, 12)
# print(y1.shape) #(993,)




x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, x2, y1, y2, train_size=0.7, shuffle= False)#random_state=123 )




#2. 모델구성

#x_1
input1 = Input(shape=(12,12),name= 'in1')
dense1 = Dense(512, activation= 'relu', name = '1')(input1)
dense2 = Dense(256, activation= 'relu', name= '2')(dense1)
dense3 = Dense(128, activation= 'relu', name = '3')(dense2)
output1 = Dense(64, activation= 'relu', name= 'out1')(dense3)


#x_2
input11 = Input(shape=(12,12), name= 'in11')
dense11 = Dense(512, activation= 'relu', name= '11' )(input11)
dense12 = Dense(256, activation= 'relu', name= '12')(dense11)
dense13 = Dense(128, activation= 'relu', name= '13')(dense12)
output11 = Dense(64, activation= 'relu', name= 'out11')(dense13)


#y_1
merge1 = concatenate([output1,output11]) 
c1 = Conv1D(20, kernel_size=3)(merge1)
l1 = LSTM(20 )(c1)
merge2 = Dense(20)(l1)
merge3 = Dense(20)(merge2)
merge4 = Dense(20)(merge3)
last_output1 = Dense(1)(merge4)
last_output2 = Dense(1)(merge4)




model = Model(inputs= [input1, input11], outputs= [last_output1, last_output2])

#model.summary()



#3. 모델, 컴파일
es = EarlyStopping(monitor='val_loss' , mode = 'auto' , patience= 300 , restore_best_weights=True , verbose= 1  )


model.compile(loss= 'mae',optimizer= 'adam', metrics= 'mae')
hist = model.fit([x1_train,x2_train], [y1_train,y2_train], epochs=500, batch_size=300, validation_split=0.2, verbose=2, callbacks=[es])

model.save_weights("c:\\_data\\sihum\\sihum_save_weights1.h5")
model.save("c:\\_data\\sihum\\save_model.h5")

# print(last_output1.shape)
# print(x1_test.shape)
# print(y1_test.shape)
# print(y1_train.shape)




#4. 평가, 예측
results = model.evaluate([x1_test, x2_test], [y1_test, y2_test])
y_predict = model.predict([x1_test,x2_test])
#r2_1 = r2_score(y1_test, y_predict[0])
#r2_2 = r2_score(y2_test, y_predict[1])
y_predict = model.predict([x1_test[0].reshape(1,12,12), x2_test[0].reshape(1,12,12)])


print(x1_test[0]) 

print(y_predict) #[array([[0.7619723]], dtype=float32), array([[0.6026146]], dtype=float32)]
print(x1.shape) #(986, 12, 12)


y1_temp = np.zeros([1,12])
print(y1_temp.shape)


y1_temp[0][0] = y_predict[0]

y_predict2 = mms1.inverse_transform(y1_temp)

print(y_predict2)




y2_temp = np.zeros([1,12])
y2_temp[0][0] = y_predict[1]
y_predict22 = mms2.inverse_transform(y2_temp)


#print('===================')
#print('삼성r2:', r2_1)
#print('아모레r2:', r2_2)


print('===================')
print("삼성시가:",np.around(y_predict2[0][0],2))
print("아모레종가:",np.around(y_predict22[0][0],2))


import matplotlib.pyplot as plt

plt.figure(figsize= (9,6))
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss', marker = '.')

plt.legend(loc = 'upper right')
plt.title("samsung LOSS")
plt.xlabel('epoch')
plt.grid()
plt.show()