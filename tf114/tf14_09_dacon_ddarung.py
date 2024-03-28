import tensorflow as tf
tf.compat.v1.set_random_seed(123)

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score


import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import time
# 1. 데이터
path = "c:\\_data\\dacon\\ddarung\\"
train_csv=pd.read_csv(path + "train.csv",index_col=0)
test_csv=pd.read_csv(path + "test.csv",index_col=0)
submission_csv =pd.read_csv(path +"submission.csv")
train_csv=train_csv.dropna()
test_csv=test_csv.fillna(test_csv.mean())
x=train_csv.drop(['count'],axis=1)
y=train_csv['count']
print(x.shape, y.shape) # (1328, 9) (1328,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle=True, random_state=123)
y_train = y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)
xp = tf.compat.v1.placeholder(tf.float32,shape=[None,9])
yp = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([9,1], dtype=tf.float32,name='weights'))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], dtype=tf.float32,name='bias'))

hypothesis = tf.compat.v1.matmul(xp, w) + b

loss = tf.reduce_mean(tf.compat.v1.square(hypothesis-yp))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.08)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 15001
for step in range(epochs):
    cost_val,_ = sess.run([loss,train],
                          feed_dict={xp:x_train,yp:y_train})
    if step %100 == 0 :
        print(step, "loss:",cost_val)
        
        from sklearn.metrics import r2_score
predict = sess.run(hypothesis, feed_dict={xp: x_test})
r2 = r2_score(y_test, predict)
print("R2:", r2)
        
sess.close()
