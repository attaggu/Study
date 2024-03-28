import tensorflow as tf
tf.compat.v1.set_random_seed(123)

# 1. Session() // sess.run(변수)


# 1. data
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)


sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(w), sess.run(b))

# 2. model
hypothesis = x*w + b
# y = x*w + b 
 
# 3. compile
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08)
train = optimizer.minimize(loss)
# model.compile(loss='mse', optimizer='sgd')    sgd = stochastic gradient descent

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # model fit
    epochs = 101
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
        if step % 20 == 0 or step == epochs -1 :
            print(step, loss_val, w_val, b_val)
sess.close()

# 2. Session( // 변수.eval(session=sess)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
wb = w.eval(session=sess)
bb = b.eval(session=sess)
print('wb', wb)
print('bb', bb)
sess.close()

# 3. InteractiveSession() // 변수.eval()

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
wc = w.eval()
bc = b.eval()
print('wc', wc)
print('bc', bc)
sess.close()



