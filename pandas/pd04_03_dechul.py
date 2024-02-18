import pandas as pd
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical

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
le = LabelEncoder()
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


x = x.astype('float32')
test_csv = test_csv.astype('float32')

def outliers(data, threshold=3):
    z_scores = np.abs((data - data.mean()) / data.std())
    outlier_indices = np.where(z_scores > threshold)
    data.iloc[outlier_indices] = data.mean()  # 이상치를 평균값으로 대체
    return data

# 이상치 처리 적용
for column in x.columns:
    x[column] = outliers(x[column])
    test_csv[column] = outliers(test_csv[column])



x_train, x_test, y_train, y_test = train_test_split(
                                                    x,
                                                    y,             
                                                    train_size=0.78,                                                   
                                                    random_state=66101,
                                                    stratify=y,
                                                    shuffle=True,
                                                    )

from sklearn.preprocessing import StandardScaler, RobustScaler 

scaler=StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


model = Sequential()
model.add(Dense(13, input_dim=13, activation='swish'))
model.add(Dense(37, activation='swish')) 
model.add(Dense(13, activation='swish'))
model.add(Dense(31, activation='swish'))
model.add(Dense(13, activation='swish'))
model.add(Dense(41, activation='swish'))
model.add(Dense(11, activation='swish'))
model.add(Dense(37, activation='swish'))
model.add(Dense(17, activation='swish'))
model.add(Dense(37, activation='swish'))
model.add(Dense(19, activation='swish'))
model.add(Dense(39, activation='swish'))
model.add(Dense(13, activation='swish'))
model.add(Dense(41, activation='swish'))
model.add(Dense(19, activation='swish'))
model.add(Dense(37, activation='swish'))
model.add(Dense(11, activation='swish'))
model.add(Dense(47, activation='swish'))
model.add(Dense(17, activation='swish'))
model.add(Dense(7, activation='softmax'))


import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
MCP_path = "../_data/_save/MCP/"
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([MCP_path, 'k23_', date, '_', filename])

hist=model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss',
                mode='auto',
                patience=200,
                verbose=1,
                restore_best_weights=True
                )
mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath=filepath,
                      )

model.fit(x_train, y_train, epochs=200, batch_size = 1302,
                validation_split=0.12,  #
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
submission_csv.to_csv(path + "submissionbb_0202_1.csv", index=False)

# 로스 :  0.44691532850265503
# acc :  0.8413028120994568
# f1 :  0.8112846280598818

# 로스 :  0.47607654333114624
# acc :  0.8338447213172913
# f1 :  0.7608123934208394