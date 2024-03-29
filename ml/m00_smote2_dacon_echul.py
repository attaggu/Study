import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical #
from imblearn.over_sampling import SMOTE

path = "C:\\_data\\dacon\\dechul\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0 )
print(train_csv.shape)  
test_csv = pd.read_csv(path + "test.csv", index_col=0 )
print(test_csv.shape) 
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv.shape)  
train_csv = train_csv[train_csv['주택소유상태'] != 'ANY']
test_csv.loc[test_csv['대출목적'] == '결혼' , '대출목적'] = '기타'
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder() # 대출기간, 대출목적, 근로기간, 주택소유상태 // 라벨 인코더 : 카테고리형 피처를 숫자형으로 변환
train_csv['주택소유상태'] = le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = le.fit_transform(train_csv['대출목적'])
train_csv['대출기간'] = train_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
train_csv['근로기간'] = le.fit_transform(train_csv['근로기간'])

test_csv['주택소유상태'] = le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = le.fit_transform(test_csv['대출목적'])
test_csv['대출기간'] = test_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
test_csv['근로기간'] = le.fit_transform(test_csv['근로기간'])

train_csv['대출등급'] = le.fit_transform(train_csv['대출등급'])

x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']

# y = np.reshape(y, (-1,1)) 

# ohe = OneHotEncoder(sparse = False)
# ohe.fit(y)
# y_ohe = ohe.transform(y)

x_train, x_test, y_train, y_test = train_test_split(
                                                    x,
                                                    y,             
                                                    train_size=0.82,                                                    
                                                    random_state=2026,
                                                    stratify=y,
                                                    shuffle=True,
                                                    )


smote=SMOTE(random_state=123, k_neighbors=3)
x_train,y_train=smote.fit_resample(x_train,y_train)
# print(np.unique(y_train,return_counts=True))

# print(pd.value_counts(y_train))


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler 

scaler = StandardScaler() # 클래스 정의

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


model = Sequential()
model.add(Dense(13, input_dim=13, activation='swish'))
model.add(Dense(61, activation='swish')) 
model.add(Dense(12, activation='swish'))
model.add(Dense(43, activation='swish'))
model.add(Dense(13, activation='swish'))
model.add(Dense(7, activation='swish'))
model.add(Dense(41, activation='swish'))
model.add(Dense(7, activation='swish'))
model.add(Dense(29, activation='swish'))
model.add(Dense(11, activation='swish'))
model.add(Dense(31, activation='swish'))
model.add(Dense(7, activation='swish'))
model.add(Dense(47, activation='swish'))
model.add(Dense(17, activation='swish'))
model.add(Dense(7, activation='softmax'))


import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
MCP_path = "../_data/_save/MCP/"
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([MCP_path, 'k23_', date, '_', filename])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=50000,
                verbose=1,
                restore_best_weights=True
                )
mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath=filepath,
                      )

model.fit(x_train, y_train, epochs=10000, batch_size = 2090 ,
                validation_split=0.08,  #
                callbacks=[es, mcp],
                verbose=1
                )

results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)  
# arg_test = np.argmax(y_test, axis=1)
y_submit = model.predict(test_csv)
submit = np.argmax(y_submit, axis=1)
submitssion = le.inverse_transform(submit)
      
submission_csv['대출등급'] = submitssion
# y_predict = ohe.inverse_transform(y_predict)
# y_test = ohe.inverse_transform(y_test)
f1 = f1_score(y_test, y_predict, average='macro')
acc = accuracy_score(y_test, y_predict)
print("로스 : ", results[0])  
print("acc : ", results[1])  
print("f1 : ", f1)  
submission_csv.to_csv(path + "submission_0126_3.csv", index=False)


# 로스 :  0.23312613368034363  
# acc :  0.9373485445976257    
# f1 :  0.9129514893151703 
