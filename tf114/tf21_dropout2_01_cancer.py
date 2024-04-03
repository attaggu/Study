from sklearn.datasets import load_breast_cancer
import tensorflow as tf
tf.compat.v1.set_random_seed(123)
from sklearn.preprocessing import StandardScaler
# data
datasets = load_breast_cancer()

x_data,y_data = datasets.data, datasets.target

print(x_data.shape, y_data.shape)   #(569, 30) (569,)
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)
y_data = y_data.reshape(-1,1)
keep_prob = tf.compat.v1.placeholder(tf.float32)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w1 = tf.compat.v1.Variable(tf.random_normal([30,10], name='weight1')) # (n,10) = (n,2)에 * (2,10)이 곱해져야 나옴
b1 = tf.compat.v1.Variable(tf.zeros([10], name='bias'))
layer1 = tf.compat.v1.matmul(x, w1) + b1        #(N,10)
layer1 = tf.compat.v1.nn.dropout(layer1, keep_prob)

# layer2 : model.add(Dense(9))
w2 = tf.compat.v1.Variable(tf.random_normal([10,9], name='weight2')) # (n,10) 에 (10,9)를 곱해 (n,9)
b2 = tf.compat.v1.Variable(tf.zeros([9], name='bias'))
layer2 = tf.compat.v1.matmul(layer1,w2) + b2    #(N, 9)
layer2 = tf.compat.v1.nn.dropout(layer2, keep_prob)

# layer3 : model.add(Dense(8))
w3 = tf.compat.v1.Variable(tf.random_normal([9,8], name='weight3')) # (n,9) 에 (9,8)를 곱해 (n,8)
b3 = tf.compat.v1.Variable(tf.zeros([8], name='bias'))
layer3 = tf.compat.v1.matmul(layer2,w3) + b3    #(N, 8)
layer3 = tf.compat.v1.nn.dropout(layer3, keep_prob)

# layer4 : model.add(Dense(7))
w4 = tf.compat.v1.Variable(tf.random_normal([8,7], name='weight4')) # (n,9) 에 (9,8)를 곱해 (n,8)
b4 = tf.compat.v1.Variable(tf.zeros([7], name='bias'))
layer4 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer3,w4) + b4)    #(N, 7)

# output_layer : model.add(Dense(1), activation='sigmoid)
w5 = tf.compat.v1.Variable(tf.random_normal([7,1], name='weight5')) # (n,9) 에 (9,8)를 곱해 (n,8)
b5 = tf.compat.v1.Variable(tf.zeros([1], name='bias'))
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer4,w5) + b5)
# model
# 순수 log 가 0이 안나오도록 조절

# compile
loss = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))  # binary_crossentropy
train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
y_predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(y,y_predict),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        cost_val, _ = sess.run([loss,train], feed_dict={x:x_data, y:y_data, keep_prob:0.8})
        
        if step % 200 ==0:
            print(step, cost_val)
            
    hypo, pred, acc= sess.run([hypothesis, y_predict, acc],
                                   feed_dict={x:x_data, y:y_data, keep_prob:1.0})

    print("훈련값 :", hypo)
    print("예측값 :", pred)
    print("acc :", acc)
