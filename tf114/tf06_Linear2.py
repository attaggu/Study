import tensorflow as tf
tf.set_random_seed(123)

# 1. Data
x = [1,2,3,4,5]
y = [1,2,3,4,5]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

hypothesis = x* w + b


loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)


sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

epochs = 5000
for step in range(epochs):
    sess.run(train)
    if step % 50 == 0:
        print(step, sess.run(loss),sess.run(w), sess.run(b))
        
sess.close()



