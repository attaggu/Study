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
                patience=19999,
                verbose=1,
                restore_best_weights=True
                )
mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      verbose=1,
                      save_best_only=True,
                      filepath=filepath,
                      )

model.fit(x_train, y_train, epochs=3333, batch_size = 1302,
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

# 로스 :  0.16194483637809753
# acc :  0.9443474411964417
# f1 :  0.9321996346622266

로스 :  0.1596727967262268
acc :  0.9465187788009644
f1 :  0.9357687099935416