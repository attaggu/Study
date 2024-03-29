
import tensorflow as tf
tf.compat.v1.set_random_seed(123)
# data
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]] # (6,2)
y_data = [[0], [0], [0], [1], [1], [1]] # (6,1)


x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1], dtype=tf.float32, name='weight'))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1], dtype=tf.float32, name='bias'))

# model
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w) + b) #sigmoid 로 감싸줌

# compile
# loss = tf.reduce_mean(tf.compat.v1.square(hypothesis-y))    # mse
loss = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis)) #변경

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
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
# [[-0.5912015 ] [-0.53942686]]       
print(type(w_val))
# <class 'numpy.ndarray'> tensorflow data 는 numpy

# evaluate | predict
x_test = tf.compat.v1.placeholder(tf.float32,shape=[None,2])
# y_predict = x_test * w_val + b_val - matmul 해야됨
y_pred = tf.compat.v1.sigmoid(tf.matmul(x_test, w_val) + b_val)
y_predict =sess.run(tf.cast(y_pred>0.5,dtype=tf.float32), feed_dict={x_test:x_data})
#tf.cast로 round 작업
print(y_predict)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data, y_predict)
print("acc:", acc)  #mse: 16.541481777213356

sess.close()
