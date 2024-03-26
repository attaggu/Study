import tensorflow as tf
tf.set_random_seed(777)


x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32) 




hypothesis = x * w + b



loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0823)
train = optimizer.minimize(loss)

# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epochs = 101
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
        #안보려고 _ 를 줌
        if step % 20 == 0:  # epochs 20마다 보여줌 - 너무 많이 떠서 = verbose
            
            print(step, loss_val, w_val, b_val)
         

    # x_predict_data = [6,7,8]

    # x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
    # #python 방식
    # y_predict = x_predict_data * w_val + b_val
    # print('[6,7,8]의 예측 :', y_predict)
    # print('==========================')
    # y_predict2 = x_test * w_val + b_val
    # print('[6,7,8]의 예측 :', sess.run(y_predict2, feed_dict={x_test : x_predict_data}))
    

    
    