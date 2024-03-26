import tensorflow as tf
tf.set_random_seed(123)

# 1. Data
x = [1,2,3]
y = [1,2,3]

w = tf.Variable(123, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

# 2. Model
# y = w*x + b -> y = x*w + b
# y = hypothesis = predict 라고 생각
hypothesis = x * w + b

# 3-1. Compile
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)
# 아래와 똑같음 
# model.compile(loss='mse', optimizer='sgd')

# 3-2. Fit
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
# 변수 초기화

# 3-3. model.fit
epochs = 10000
for step in range(epochs):
    sess.run(train) # 핵심 - 1 epoch
    if step % 20 == 0:  # epochs 20마다 보여줌 - 너무 많이 떠서 = verbose
        print(step, sess.run(loss), sess.run(w), sess.run(b))
        # verbose 와 model.weight 에서 확인가능한 값들
    
sess.close()    # sess가 메모리에 남아있어서 끝난뒤 꺼줘야함
