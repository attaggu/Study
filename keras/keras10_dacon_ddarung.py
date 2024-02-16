# https://dacon.io/competitions/open/235576/overview/description
#setting
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error #rmse 정의 하기 위해


#1.data

path = "c:\\_data\\dacon\\ddarung\\"
# print(path + "aaa.csv")  #c:\_data\dacon\ddarung\aaa.csv
#/, //, \, \\ 다 된다

# train_csv = pd.read_csv("c:\_data\dacon\\ddarung\\train.csv")
train_csv = pd.read_csv(path + "train.csv",index_col=0) #0=첫번째 id 컬럼은 데이터가 아니여서 인덱스로 지정한다.
print(train_csv)
test_csv = pd.read_csv(path + "test.csv",index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path + "submission.csv")
print(submission_csv) 

print(train_csv.shape)  #(1459, 10) 11 에서 id가 빠져서 10 / 10번째꺼는 count=y
print(test_csv.shape)   #(715, 9) 10에서 id가 빠져서 9
print(submission_csv.shape) #(715, 2)---아이디 2개 중복

print(train_csv.columns)    
    #(['id'---아이디가빠짐, 'hour', 'hour_bef_temperature', 'hour_bef_precipitation',
    #    'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
    #    'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count']
print(train_csv.info())
print(test_csv.info())

print(train_csv.describe()) #describe는 함수라 가로 표시 , 데이터에 대한 설명
print(test_csv.describe())

########### 결측치 처리 1. 제거 ##########
print(train_csv.isna().sum()) # isna = isnull
train_csv = train_csv.dropna()  # 결측치 한 행에 하나라도 있으면 그 행 삭제
print(train_csv.isna().sum()) # 결측치 삭제 재확인
print(train_csv.info())
print(train_csv.shape)  #(1328,10) 결측치 삭제된 값

test_csv = test_csv.fillna(test_csv.mean()) #결측치 컬럼 평균을 넣음
print(test_csv.info())


######### x와 y를 분리########### ---0이나 평균도 넣어봄
x = train_csv.drop(['count'],axis=1)    # axis는 축 0=행, 1 = 열 = count는 행 => train_csv에서 count컬럼 제거
print(x)   
y = train_csv['count']   #train_csv에서 count컬럼만 가져온다
print(y)
x_train,x_test,y_train,y_test = train_test_split(
    x,y, shuffle=True, train_size=0.75, random_state=563
)

print(x_train.shape, x_test.shape)  #(929, 9) (399, 9)
print(y_train.shape, y_test.shape)  #(929,) (399,)

#model
model=Sequential()
model.add(Dense(9, input_dim=9))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(11))
model.add(Dense(1))


#3 compile,fit

model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train, epochs=1000, batch_size=100)

#4
loss=model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)   #(715, 1) 


############submission.csv 만들기 (count컬럼 값만 넣어주면 됨)############

submission_csv['count']=y_submit # submission 안 count 컬럼에 집어넣다
print(submission_csv)
submission_csv.to_csv(path + "submission_0105.csv", index=False) # 파일 생성(index 제거)

y_predict=model.predict(x_test)

r2 = r2_score(y_test,y_predict)
print("loss:", loss)

print("R2_score:",r2)

### train 결축 제거 , test 결축 평균 ###























