import tensorflow as tf
tf.set_random_seed(123)

x = [1,2,3]
y = [1,2,3]

w = tf.Variable(123, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

hypothesis = x * w + b

loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epochs = 10000
    for step in range(epochs):
        sess.run(train) # 핵심 - 1 epoch
        if step % 20 == 0:  # epochs 20마다 보여줌 - 너무 많이 떠서 = verbose
            print(step, sess.run(loss), sess.run(w), sess.run(b))
        # verbose 와 model.weight 에서 확인가능한 값들
    
# sess.close() => with를 사용하기 때문에 사용안해도됨

