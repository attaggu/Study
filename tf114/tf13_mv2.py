import tensorflow as tf
tf.compat.v1.set_random_seed(123)

# (5, 3)
x_data = [[73,51,65],               
          [92,98,11],
          [89,31,33],
          [99,33,100],
          [17,66,79]]
# (5, 1)
y_data = [[152],[185],[180],[205],[142]]


x = tf.compat.v1.placeholder(tf.float32, shape=[None,3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1],dtype=tf.float32, name='weight'))
# (5, 3) * (3, 1) = (5, 1)

b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1],dtype=tf.float32, name='bias'))

hypothesis = tf.matmul(x, w) + b
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis-y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1001
for step in range(epochs):
    cost_val, _ = sess.run([loss, train], 
                           feed_dict={x:x_data, y:y_data})
    if step %20 == 0 :
        print(step,"loss : ", cost_val)

from sklearn.metrics import r2_score
predict = sess.run(hypothesis, feed_dict={x: x_data})
r2 = r2_score(y_data, predict)
print("R2:", r2)
        
sess.close()