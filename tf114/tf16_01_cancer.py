from sklearn.datasets import load_breast_cancer
import tensorflow as tf
tf.compat.v1.set_random_seed(123)

# data
datasets = load_breast_cancer()

x_data,y_data = datasets.data, datasets.target

print(x_data.shape, y_data.shape)   #(569, 30) (569,)

y_data = y_data.reshape(-1,1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,1], dtype=tf.float32, name='weight'))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], dtype=tf.float32, name='bias'))

# model
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w)+b)
hypothesis2 = tf.compat.v1.clip_by_value(hypothesis, 0.0000001, 1 - 0.0000001)
# 순수 log 가 0이 안나오도록 조절

# compile
loss = -tf.reduce_mean(y*tf.log(hypothesis2) + (1-y)*tf.log(1-hypothesis2))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 10001

for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b],
                                         feed_dict={x:x_data,y:y_data})
    if step %50 == 0 :
        print(step, "loss : ", cost_val)
        
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None,30])
y_pred = tf.compat.v1.sigmoid(tf.matmul(x_test, w_val)+ b_val)
y_predict = sess.run(tf.cast(y_pred>0.5, dtype=tf.float32), feed_dict={x_test:x_data})

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data, y_predict)

print("acc : ", acc)
sess.close()