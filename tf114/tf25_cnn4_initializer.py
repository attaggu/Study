import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(123)

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

w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 1],
                               initializer=tf.contrib.layers.xavier_initializer())
# 2,2 = 커널사이즈 / 1 = 컬러값(채널) / 64 = 아웃풋(필터)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    w1_val = sess.run(w1)
    print(w1_val,'\n', w1_val.shape)

# [[[[-0.07556218]]

#   [[ 0.24956447]]]


#  [[[-0.04803634]]

#   [[-0.04819453]]]]
# (2, 2, 1, 64)

'''
b1 = tf.compat.v1.Variable(tf.zeros([64]), name='b1')
# bias 개수는 아웃풋(필터) 개수와 동일

L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='VALID')
# 4차원이라 stride=[1,'1','1',1] 가운데 1,1이 stride / 앞뒤 1,1 은 shape 맞춰주기용
# model.add(Conv2d(64,kenel_size=(2,2), strides=(1,1), input_shape=(28,28,1)))

L1 = L1 + b1   # L1 += b1
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
# 4차원이라 앞뒤 1 / 1,1을 주면 값이 그대로라 의미없음 (n,n)안에서 가장 큰 값만 뽑아 쓰기 때문

print(L1)   # Tensor("Relu:0", shape=(?, 27, 27, 64), dtype=float32)  - 렐루 적용
print(L1_maxpool)   
# VALID = Tensor("MaxPool2d:0", shape=(?, 13, 13, 64), dtype=float32)  
# SAME = Tensor("MaxPool2d:0", shape=(?, 14, 14, 64), dtype=float32) 

print(w1)   #<tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1)   #Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32) 

w2 = tf.compat.v1.get_variable('w2', shape=[3,3, 64, 32])
# 3,3 = 커널사이즈 / 64 = 인풋(이전 아웃풋 = 이미지 장수) / 32 = 아웃풋(필터)
# 인풋 - 이미지 장수를 맞춰야함

L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')
'''
