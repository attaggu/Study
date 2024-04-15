import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(123)

# 1. data

x_train = np.array([[ [[1],[2],[3]],
                      [[4],[5],[6]],
                      [[7],[8],[9]] ]])
print(x_train.shape)    #(1, 3, 3, 1)

x = tf.compat.v1.placeholder(tf.float32, [None,3,3,1])
w = tf.compat.v1.constant([[[[1.]],[[0.]]],
                           [[[1.]],[[0.]]]])
print(w)    # (2, 2, 1, 1)  =  2,2 커널 / 1 채널 / 1 아웃풋필터

L1 = tf.nn.conv2d(x, w, strides=(1,1,1,1), padding='VALID')
print(L1)   # (?, 2, 2, 1)

sess = tf.compat.v1.Session()
output = sess.run(L1, feed_dict={x:x_train})

print("============결과===============")
print(output)
print("=============결과shape==============")
print(output.shape)
# (1, 2, 2, 1) = 1장 2,2,1 사진
