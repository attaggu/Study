

import tensorflow as tf
import numpy as np


# 1. data
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
tf.set_random_seed(777)

# 1. 데이터
x_data, y_datas = fetch_covtype(return_X_y=True)
ys= y_datas.reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)
print(x_data.shape, ys.shape)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe=OneHotEncoder(sparse=False)
y_data=ohe.fit_transform(ys)
print(x_data.shape,y_data.shape)
# (581012, 54) (581012, 7)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,54])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,7])

w1 = tf.compat.v1.Variable(tf.random_normal([54,10], name='weight1')) # (n,10) = (n,2)에 * (2,10)이 곱해져야 나옴
b1 = tf.compat.v1.Variable(tf.zeros([10], name='bias1'))
layer1 = tf.compat.v1.matmul(x, w1) + b1        #(N,10)

# layer2 : model.add(Dense(9))
w2 = tf.compat.v1.Variable(tf.random_normal([10,9], name='weight2')) # (n,10) 에 (10,9)를 곱해 (n,9)
b2 = tf.compat.v1.Variable(tf.zeros([9], name='bias2'))
layer2 = tf.compat.v1.matmul(layer1,w2) + b2    #(N, 9)

# layer3 : model.add(Dense(8))
w3 = tf.compat.v1.Variable(tf.random_normal([9,8], name='weight3')) # (n,9) 에 (9,8)를 곱해 (n,8)
b3 = tf.compat.v1.Variable(tf.zeros([8], name='bias3'))
layer3 = tf.compat.v1.matmul(layer2,w3) + b3    #(N, 8)

# layer4 : model.add(Dense(7))
w4 = tf.compat.v1.Variable(tf.random_normal([8,7], name='weight4')) # (n,9) 에 (9,8)를 곱해 (n,8)
b4 = tf.compat.v1.Variable(tf.zeros([7], name='bias4'))
layer4 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer3,w4) + b4)    #(N, 7)

# output_layer : model.add(Dense(1), activation='sigmoid)
w5 = tf.compat.v1.Variable(tf.random_normal([7,7], name='weight5')) # (n,9) 에 (9,8)를 곱해 (n,8)
b5 = tf.compat.v1.Variable(tf.zeros([7], name='bias5'))

hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer4,w5) + b5)


# 2. model
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer4,w5) + b5)

# 3-1. compile
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis + 1e-5),axis=1))  #categorical

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-4)
# train = optimizer.minimize(loss)
# ===똑같음
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 101
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss,train,w5,b5],
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


