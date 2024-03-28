import tensorflow as tf
tf.compat.v1.set_random_seed(123)
import matplotlib.pyplot as plt

# 1. data
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# 2. model
hypothesis = x*w + b

# 3. compile
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.008)
train = optimizer.minimize(loss)


# 3. train
loss_val_list = []
w_val_list = []
b_val_list = []

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    epochs = 101
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
        if step % 20 == 0 or step == epochs-1 :
            print(step, loss_val, w_val, b_val)
        
        loss_val_list.append(loss_val)
        w_val_list.append(w_val)
        b_val_list.append(b_val)

# history 리스트 찍어보기
print(loss_val_list)
print('')
print(w_val_list)

# val_loss 시각화
plt.subplot(2,2,1)
plt.plot(loss_val_list)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('val_loss visualize')

# weights 시각화
plt.subplot(2,2,2)
plt.plot(w_val_list)
plt.xlabel('epochs')
plt.ylabel('weights')
plt.title('weights visualize')

# bias 시각화
plt.subplot(2,2,3)
plt.scatter(range(len(b_val_list)), b_val_list)
plt.xlabel('epochs')
plt.ylabel('bias')
plt.title('bias visualize')

# val_loss와 weights의 관계 시각화
plt.subplot(2,2,4)
plt.scatter(w_val_list, loss_val_list)
plt.xlabel('weights')
plt.ylabel('loss')
plt.title('relations of val_loss weights visualize')


plt.tight_layout()
plt.show()