import tensorflow as tf
tf.compat.v1.set_random_seed(123)

# 1. data

x_data = [[0,0], [0,1], [1,0], [1,1]]   # (4,2)
y_data = [[0], [1], [1], [0]]           # (4,1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])
w = tf.compat.v1.Variable(tf.random_normal([2,1], name='weight'))
b = tf.compat.v1.Variable(tf.zeros([1], name='bias'))

hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w) + b)


loss = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1001

for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss,train,w,b],
                           feed_dict={x:x_data, y:y_data})
    if step %20 == 0 :
        print(step, "loss : ", cost_val)
        print(w_val)

x_test = tf.compat.v1.placeholder(tf.float32,shape=[None,2])
pred = tf.compat.v1.sigmoid(tf.matmul(x_test,w_val)+b_val)
y_predict = sess.run(tf.cast(pred > 0.5, dtype=tf.float32), feed_dict={x_test:x_data})
from sklearn.metrics import accuracy_score, mean_squared_error
acc = accuracy_score(y_data,y_predict)
print("acc:", acc)   
