
import tensorflow as tf
import numpy as np


# 1. data
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
tf.set_random_seed(777)

# 1. 데이터
x_data, y_datas = load_digits(return_X_y=True)
ys= y_datas.reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe=OneHotEncoder(sparse=False)
y_data=ohe.fit_transform(ys)
print(x_data.shape,y_data.shape)
# (1797, 64) (1797, 10)
x = tf.compat.v1.placeholder(tf.float32, shape=[None,64])
w = tf.compat.v1.Variable(tf.random_normal([64,10]), name='weight')   # None,4 - None,3
b = tf.compat.v1.Variable(tf.zeros([1,10]), name='bias') #4,3 - None,3
y = tf.compat.v1.placeholder(tf.float32, shape=[None,10])

# 2. model
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x,w) + b)

# 3-1. compile
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis + 1e-5),axis=1))  #categorical

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-4)
# train = optimizer.minimize(loss)
# ===똑같음
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.4).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1001
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


