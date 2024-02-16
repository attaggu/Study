from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "c://_data//dacon//dechul//"

train_csv=pd.read_csv(path+"train.csv",index_col=0)

test_csv=pd.read_csv(path+"test.csv",index_col=0)

sub_csv=pd.read_csv(path+"sample_submission.csv")

# train_csv=train_csv[train_csv['근로기간'] != 'Unknown']

# unique,count = np.unique(train_csv['근로기간'], return_counts=True)

le=LabelEncoder()
train_le=LabelEncoder()
test_le=LabelEncoder()




train_csv['대출기간']=train_le.fit_transform(train_csv['대출기간'])
test_csv['대출기간']=test_le.fit_transform(test_csv['대출기간'])
train_csv['근로기간']=train_le.fit_transform(train_csv['근로기간'])
train_csv['주택소유상태']=train_le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적']=train_le.fit_transform(train_csv['대출목적'])
train_csv['대출등급']=train_le.fit_transform(train_csv['대출등급'])

test_csv['근로기간']=test_le.fit_transform(test_csv['근로기간'])
test_csv['주택소유상태']=test_le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적']=test_le.fit_transform(test_csv['대출목적'])


# train_csv['대출기간'] = train_csv['대출기간'].str.split().str[0].astype(int)
# test_csv['대출기간'] = test_csv['대출기간'].str.split().str[0].astype(int)
# train_csv['대출기간']=train_le.fit_transform(train_csv['대출기간'])
# test_csv['대출기간']=test_le.fit_transform(test_csv['대출기간'])





x=train_csv.drop('대출등급',axis=1)
y=train_csv['대출등급']

train_csv.dropna

# print(np.unique(y)) #['A' 'B' 'C' 'D' 'E' 'F' 'G']
# print(pd.value_counts(y))
# B    28817
# C    27623
# A    16772
# D    13354
# E     7354
# F     1954
# G      420

# print(train_csv.head(8))
yo = to_categorical(y)

print(x.shape,yo.shape) #(96294, 13) (96294, 7)

x_train,x_test,y_train,y_test=train_test_split(x,yo,train_size=0.8,
                                               random_state=112,stratify=yo)

model=Sequential()
model.add(Dense(1,input_shape=(13,)))
model.add(Dense(10,))
model.add(Dense(10,))
model.add(Dense(10,))
model.add(Dense(10,))
model.add(Dense(10,))
model.add(Dense(10,))
model.add(Dense(7,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es=EarlyStopping(monitor='val_acc',mode='min',patience=2000,verbose=1,
                 restore_best_weights=True)
hist=model.fit(x_train,y_train,epochs=4000,batch_size=500,validation_split=0.2,callbacks=[es])

result=model.evaluate(x_test,y_test)
print("loss",result[0])
print("acc",result[1])
y_predict=model.predict(x_test)

arg_y_test=np.argmax(y_test,axis=1)
arg_y_predict=np.argmax(y_predict,axis=1)
f1_score=f1_score(arg_y_test,arg_y_predict,average='weighted')
print("f1_score:",f1_score)
y_submit=np.argmax(model.predict(test_csv),axis=1)
y_submit=train_le.inverse_transform(y_submit)


sub_csv['대출등급']=y_submit
sub_csv.to_csv(path+"sample_submission_2.csv",index=False)
