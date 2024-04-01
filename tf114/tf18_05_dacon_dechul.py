



import tensorflow as tf
import numpy as np


# 1. data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import OneHotEncoder, StandardScaler
warnings.filterwarnings('ignore')
tf.set_random_seed(777)

# 1. 데이터

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

x_data = train_csv.drop(['대출등급'], axis=1)
y_datas = train_csv['대출등급']
from sklearn.preprocessing import OneHotEncoder, StandardScaler

scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)
ys = y_datas.values.reshape(-1,1)
print(x_data.shape, ys.shape)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe=OneHotEncoder(sparse=False)
y_data=ohe.fit_transform(ys)
print(x_data.shape,y_data.shape)
# (96293, 13) (96293, 7)
x = tf.compat.v1.placeholder(tf.float32, shape=[None,13])
w = tf.compat.v1.Variable(tf.random_normal([13,7]), name='weight')   # None,4 - None,3
b = tf.compat.v1.Variable(tf.zeros([1,7]), name='bias') #4,3 - None,3
y = tf.compat.v1.placeholder(tf.float32, shape=[None,7])

# 2. model
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x,w) + b)

# 3-1. compile
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis + 1e-9),axis=1))  #categorical

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-4)
# train = optimizer.minimize(loss)
# ===똑같음
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.005).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 10001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss,train,w,b],
                                         feed_dict={x:x_data, y:y_data})
    if step %20 == 0:
        print(step, "loss : ", cost_val)


pred = sess.run(hypothesis, feed_dict={x:x_data})
print(pred) # 8행 3열
pred = sess.run(tf.argmax(pred,axis=1))
print(pred)
y_data = np.argmax(y_data, axis=1)
print(y_data)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data,pred)
print("acc : ", acc)
sess.close()

