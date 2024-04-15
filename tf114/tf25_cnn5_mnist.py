import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(123)
tf.compat.v1.disable_eager_execution()
# 1. data

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float')/255.
 
# 2. model
x = tf.compat.v1.placeholder(tf.float32, [None,28,28,1])    # input_shape
y = tf.compat.v1.placeholder(tf.float32, [None,10])

w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 128],)
                            #    initializer=tf.contrib.layers.xavier_initializer())
# 2,2 = 커널사이즈 / 1 = 컬러값(채널) / 64 = 아웃풋(필터)

b1 = tf.compat.v1.Variable(tf.zeros([128]), name='b1')
# bias 개수는 아웃풋(필터) 개수와 동일

L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='VALID')
# 4차원이라 stride=[1,'1','1',1] 가운데 1,1이 stride / 앞뒤 1,1 은 shape 맞춰주기용
# model.add(Conv2d(64,kenel_size=(2,2), strides=(1,1), input_shape=(28,28,1)))

L1 = L1 + b1   # L1 += b1
L1 = tf.nn.relu(L1)
# L1 = tf.nn.dropout(L1, keep_prob=0.7)
# L1 = tf.nn.dropout(L1, rate=0.3) 같다 = model.add(Dropout(0.3))
# dropout 0.7을 살린다 - 30% 자른다
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
# 4차원이라 앞뒤 1 / 1,1을 주면 값이 그대로라 의미없음 (n,n)안에서 가장 큰 값만 뽑아 쓰기 때문


w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 128, 64],)
                            #    initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.compat.v1.Variable(tf.zeros([64]), name='b2')


L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='SAME')

L2 = L2 + b2   # L1 += b1
L2 = tf.nn.relu(L2)
# L2 = tf.nn.dropout(L2, keep_prob=0.9)
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
print(L2)
print(L2_maxpool)


w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 64, 32],)
                            #    initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.compat.v1.Variable(tf.zeros([32]), name='b3')

L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='SAME')
L3 = L3 + b3
L3 = tf.nn.relu(L3)
# L3 = tf.nn.dropout(L3, keep_prob=0.9)
# L3_maxpool = tf.nn.max_pool2d(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
print(L3)
# Tensor("Relu_2:0", shape=(?, 6, 6, 32), dtype=float32)

# Flatten

L_flat = tf.reshape(L3, [-1, 6*6*32])
print("flatten", L_flat)    # flatten Tensor("Reshape:0", shape=(?, 1152), dtype=float32)

# Layer4 DNN

w4 = tf.compat.v1.get_variable('w4', shape=[6*6*32, 100],)
                            #    initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.compat.v1.Variable(tf.zeros([100], name='b4'))
L4 = tf.nn.relu(tf.matmul(L_flat, w4) + b4)
L4 = tf.nn.dropout(L4, rate=0.3)

# Layer5 DNN

w5 = tf.compat.v1.get_variable('w5', shape=[100,10],)
                            #    initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.compat.v1.Variable(tf.zeros([10], name='b5'))
L5 = tf.nn.relu(tf.matmul(L4, w5) + b5)
hypothesis = tf.nn.softmax(L5)

loss = tf.compat.v1.losses.softmax_cross_entropy(y, hypothesis)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.005).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

training_epochs = 60
batch_size = 100


total_batch = int(len(x_train)/batch_size)
print(total_batch)

for step in range(training_epochs):
    avg_cost = 0
    
    for i in range(total_batch):
        start = i*batch_size
        end = start + batch_size
        
        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        cost_val,_,w_val,b_val = sess.run([loss,train,w5,b5],
                                          feed_dict=feed_dict)
        avg_cost +=cost_val / total_batch
    if step %20 ==0:
        print(step, "loss : ", avg_cost)

pred = sess.run(hypothesis, feed_dict={x:x_test})

argpred = sess.run(tf.math.argmax(pred,axis=1))

y_data = np.argmax(y_test, axis=1)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data, argpred)
print("acc : ", acc)
sess.close()