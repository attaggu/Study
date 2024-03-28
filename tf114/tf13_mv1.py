import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

# 1. data

x1_data = [73., 93., 89., 96., 73.]     # 국어
x2_data = [80., 88., 91., 98., 66.]     # 영어
x3_data = [75., 93., 90., 100., 70.]    # 수학
y_data = [152., 185., 180., 196., 142.] # 환산점수



x1 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x2 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x3 = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], dtype=tf.float32))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], dtype=tf.float32))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], dtype=tf.float32))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], dtype=tf.float32))

# w1 = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight1')
# w2 = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight2')
# w3 = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight3')
# b = tf.compat.v1.Variable([10], dtype=tf.float32, name='bias')

# 2. model

hypothesis = x1*w1 + x2*w2 + x3*w3 + b
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis-y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5) # 0.00001
train = optimizer.minimize(loss)

# '''
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epochs = 1001
for step in range(epochs):
    cost_val,_ = sess.run([loss, train], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    if step %20 == 0 :
        print(step, cost_val)
        
sess.close()
# '''
# with tf.compat.v1.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     epochs = 101
#     for step in range(epochs):
#         _, loss_val, w1_val, w2_val, w3_val, b_val = sess.run([train, loss, w1, w2, w3, b],
#                                                               feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
#         if step %20 == 0:
            
#             print(step, loss_val, w1_val, w2_val, w3_val, b_val)
            

