import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(123)

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])
w = tf.placeholder(tf.float32, shape=[None])

hypothesis = x*w
loss = tf.reduce_mean(tf.square(hypothesis - y))

w_values = [-30, -15, -1, 0, 1, 15, 30, 50]
x_data = [1,2]
y_data = [1,2]

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess:
    for curr_w in w_values:
        curr_loss = sess.run(loss, 
                             feed_dict={x:x_data, y:y_data, w:[curr_w for _ in x_data]})
        # w:[-30,30]
        
        w_history.append(curr_w)
        loss_history.append(curr_loss)
        
print("="*20, 'W_history', "="*20)
print(w_history)
print("="*20, 'L_history', "="*20)
print(loss_history)

plt.plot(w_history, loss_history)
plt.xlabel('Weights')
plt.ylabel('Loss')
plt.show()