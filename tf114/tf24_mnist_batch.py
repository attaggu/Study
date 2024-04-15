import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()  # gpu
tf.compat.v1.set_random_seed(777)
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape,x_test.shape)   #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape,y_test.shape)   #(60000, 10) (10000, 10)  

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]).astype('float32')/255 # 255아니면 127.5로 나눠서 정규화
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]).astype('float32')/255 # 255아니면 127.5로 나눠서 정규화


print(x_train.shape,x_test.shape)   #(60000, 784) (10000, 784)
print(y_train.shape,y_test.shape)   #(60000, 10) (10000, 10)  

x = tf.compat.v1.placeholder(tf.float32, shape=[None,784])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,10])
keep_prob = tf.compat.v1.placeholder(tf.float32)

# get_variable 안에 들어간 함수에 가중치 초기화가 한번 진행되면 다음 에포때는 적용되지 않도록 설정되어 있음
w1 = tf.compat.v1.get_variable('weight1',shape=[784,128],
                            #    initializer=tf.contrib.layers.xavier_initializer()   # 가중치 초기화 
                               ) # (n,10) = (n,2)에 * (2,10)이 곱해져야 나옴
b1 = tf.compat.v1.Variable(tf.zeros([128], name='bias1'))
layer1 = tf.compat.v1.matmul(x, w1) + b1        #(N,10)
layer1 = tf.compat.v1.nn.swish(layer1)
dropout1 = tf.compat.v1.nn.dropout(layer1, keep_prob=keep_prob)

# layer2 : model.add(Dense(9))
w2 = tf.compat.v1.get_variable('weight2',shape=[128,64],
                                # initializer=tf.contrib.layers.xavier_initializer()
                               ) # (n,10) 에 (10,9)를 곱해 (n,9)
b2 = tf.compat.v1.Variable(tf.zeros([64], name='bias2'))
layer2 = tf.compat.v1.matmul(dropout1,w2) + b2    #(N, 9)
layer2 = tf.compat.v1.nn.swish(layer2)
dropout2 = tf.compat.v1.nn.dropout(layer2, keep_prob=keep_prob)

# layer3 : model.add(Dense(8))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64,32], name='weight3')) # (n,9) 에 (9,8)를 곱해 (n,8)
b3 = tf.compat.v1.Variable(tf.zeros([32], name='bias3'))
layer3 = tf.compat.v1.matmul(dropout2,w3) + b3    #(N, 8)
layer3 = tf.compat.v1.nn.swish(layer3)
dropout3 = tf.compat.v1.nn.dropout(layer3, keep_prob=keep_prob)

# layer4 : model.add(Dense(7))
w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32,16], name='weight4')) # (n,9) 에 (9,8)를 곱해 (n,8)
b4 = tf.compat.v1.Variable(tf.zeros([16], name='bias4'))
layer4 =tf.compat.v1.matmul(dropout3,w4)+b4
# layer4 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(dropout3,w4) + b4)    #(N, 7)

# output_layer : model.add(Dense(1), activation='sigmoid)
w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16,10], name='weight5')) # (n,9) 에 (9,8)를 곱해 (n,8)
b5 = tf.compat.v1.Variable(tf.zeros([10], name='bias5'))

hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer4,w5) + b5)
# model
# 순수 log 가 0이 안나오도록 조절

# compile
# loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis + 1e-5),axis=1))  #categorical
# loss = tf.reduce_mean(-tf.reduce_sum(y*tf.compat.v1.nn.log_softmax(hypothesis), axis=1))
loss = tf.compat.v1.losses.softmax_cross_entropy(y,hypothesis)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

training_epochs = 20001
batch_size = 100

total_batch = int(len(x_train) / batch_size)
#60000 / 100
print(total_batch)  # 600.0 - 로 나와서 int 붙여줌

for step in range(training_epochs):
    
    avg_cost = 0
    
    for i in range(total_batch):
        start = i * batch_size  # 0*100 / 1*100 / 2*100 식으로 늘어남
        end = start + batch_size    # 0 + 100 / 100 + 100 / 200 + 100 식으로 배치 늘어남 
        
        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y, keep_prob:0.5}
    
        cost_val, _, w_val, b_val = sess.run([loss,train,w5,b5],
                                         feed_dict=feed_dict)
        avg_cost += cost_val / total_batch  # 600번 로스를 600번 나눠줌 - 600번 로스의 평균을 구해서 각 로스마다 값을 평가함
    # i+=1  === i+1 이랑 같음
    if step %20 == 0:
        print(step, "loss : ", avg_cost )




pred = sess.run(hypothesis, feed_dict={x:x_test,keep_prob:1.0})
print(pred) # 8행 3열
argpred = sess.run(tf.math.argmax(pred,axis=1))
print(argpred)
y_data = np.argmax(y_test, axis=1)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data,argpred)
print("acc : ", acc)
sess.close()

# acc : 0.939
