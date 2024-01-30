from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, GRU
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
# from function_package import split_x, split_xy
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = "C:\_data\kaggle/jena\\"

datasets = pd.read_csv(path+"jena.csv", index_col=0)

print(datasets.columns)
col = datasets.columns

# MinMaxScale
minmax = MinMaxScaler()
minmax_for_y = MinMaxScaler().fit(np.array(datasets['T (degC)']).reshape(-1,1))
datasets = minmax.fit_transform(datasets)
datasets = pd.DataFrame(datasets,columns=col)   # 다시 DataFrame으로, 이유는 밑의 함수들을 이용하기 위해서

# print(row_x.isna().sum(),row_y.isna().sum())    #결측치 존재하지 않음

# data RNN에 맞게 변환
def split_xy(data, time_step, y_col):
    result_x = []
    result_y = []
    
    num = len(data) - time_step                 # x만자른다면 len(data)-time_step+1이지만 y도 잘라줘야하므로 +1이 없어야함
    for i in range(num):
        result_x.append(data[i : i+time_step])  # i 부터  time_step 개수 만큼 잘라서 result_x에 추가
        y_row = data.iloc[i+time_step]          # i+time_step번째 행, 즉 result_x에 추가해준 바로 다음순번 행
        result_y.append(y_row[y_col])           # i+time_step번째 행에서 원하는 열의 값만 result_y에 추가
    
    return np.array(result_x), np.array(result_y)

x, y = split_xy(datasets,3,'T (degC)')

print("x, y: ",x.shape,y.shape)     #(420548, 3, 14) (420548,)
print(x[0],y[0],sep='\n')           #검증완료

# train test split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=False)#,random_state=333)
print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")

# model
model = Sequential()
model.add(LSTM(128, input_shape=x_train.shape[1:]))
model.add(Dense(5, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

# compile & fit
model.compile(loss='mse',optimizer='adam')
es = EarlyStopping(monitor='val_loss',mode='auto',patience=100,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=40,batch_size=8192,validation_split=0.2,verbose=2,callbacks=[es])

# model = load_model("C:\_data\KAGGLE\Jena_Climate_Dataset\model_save\\r2_0.9994.h5")

# evaluate
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)

print(f"LOSS: {loss}\nR2:  {r2}")
model.save(path+f"model_save/r2_{r2:.4f}.h5")
print(x_test.shape,y_test.shape,y_predict.shape)
# 위에서 y까지 minmax해버렸기에 inverse_transform 해주기
predicted_degC = minmax_for_y.inverse_transform(np.array(y_predict).reshape(-1,1))
y_true = minmax_for_y.inverse_transform(np.array(y_test).reshape(-1,1))
print(x_test.shape,y_true.shape,y_predict.shape)

# 실제로 잘 나온건지 원 데이터와 비교하기 위한 csv파일 생성
submit = pd.DataFrame(np.array([y_true,predicted_degC]).reshape(-1,2),columns=['true','predict'])
submit.to_csv(path+f"submit_r2_{r2}.csv",index=False)
