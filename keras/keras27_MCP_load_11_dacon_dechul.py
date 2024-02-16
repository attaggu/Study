

from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import RobustScaler,StandardScaler

path = "c://_data//dacon//dechul//"

train_csv=pd.read_csv(path+"train.csv",index_col=0)

test_csv=pd.read_csv(path+"test.csv",index_col=0)

sub_csv=pd.read_csv(path+"sample_submission.csv")

# train_csv=train_csv[train_csv['근로기간'] != 'Unknown']

# unique,count = np.unique(train_csv['근로기간'], return_counts=True)

le=LabelEncoder()


# train_csv['대출기간']=train_le.fit_transform(train_csv['대출기간'])
# test_csv['대출기간']=test_le.fit_transform(test_csv['대출기간'])
train_csv['대출기간'] = train_csv['대출기간'].str.split().str[0].astype(int)
test_csv['대출기간'] = test_csv['대출기간'].str.split().str[0].astype(int)
train_csv['근로기간']=le.fit_transform(train_csv['근로기간'])
train_csv['주택소유상태']=le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적']=le.fit_transform(train_csv['대출목적'])

test_csv['근로기간']=le.fit_transform(test_csv['근로기간'])
test_csv['주택소유상태']=le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적']=le.fit_transform(test_csv['대출목적'])


train_csv['대출등급']=le.fit_transform(train_csv['대출등급'])


x=train_csv.drop('대출등급',axis=1)
y=train_csv['대출등급']



yo = to_categorical(y)

# train_csv.dropna

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

print(x.shape,yo.shape) #(96294, 13) (96294, 7)

x_train,x_test,y_train,y_test=train_test_split(x,yo,train_size=0.9,
                                               random_state=112,stratify=yo)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


hist=model=load_model('../_data/_save/MCP/keras26_dacon_dechul_MCP.hdf5')

result=model.evaluate(x_test,y_test)
print("loss",result[0])
print("acc",result[1])
y_predict=model.predict(x_test)

arg_y_test=np.argmax(y_test,axis=1)
arg_y_predict=np.argmax(y_predict,axis=1)
f1_score=f1_score(arg_y_test,arg_y_predict,average='macro')
print("f1_score:",f1_score)
y_submit=np.argmax(model.predict(test_csv),axis=1)
y_submit=le.inverse_transform(y_submit)


sub_csv['대출등급']=y_submit
sub_csv.to_csv(path+"sample_submission_4.csv",index=False)

# scaler = MinMaxScaler()   /   f1_score: 0.3077110807495997

# scaler = StandardScaler() /   f1_score: 0.31699199191443983

# scaler = MaxAbsScaler()   /   f1_score: 0.3183024740291938

# scaler = RobustScaler()   /   f1_score: 0.3159634653253046

# loss 0.47353947162628174
# acc 0.8445482850074768
# f1_score: 0.7970325748260676