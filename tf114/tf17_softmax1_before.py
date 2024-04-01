import tensorflow as tf
import numpy as np
tf.set_random_seed(123)


# 1. data

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]

y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None,4])
w = tf.compat.v1.Variable(tf.random_normal([4,3]), name='weight')   # None,4 - None,3
b = tf.compat.v1.Variable(tf.zeros([1,3]), name='bias') #4,3 - None,3
y = tf.compat.v1.placeholder(tf.float32, shape=[None,3])

# 2. model
hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

# 3-1. compile
# loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))  #mse
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1))
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-4)
# train = optimizer.minimize(loss)
# ===똑같음
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss,train,w,b],
                                         feed_dict={x:x_data, y:y_data})
    if step %20 == 0:
        print(step, "loss : ", cost_val)

x_test = tf.compat.v1.placeholder(tf.float32, shape=[None,4])

pred = sess.run(hypothesis, feed_dict={x_test:x_data})
print(pred, sess.run(tf.argmax(pred,1)))