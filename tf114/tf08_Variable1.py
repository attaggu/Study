import tensorflow as tf
tf.compat.v1.set_random_seed(123)

변수 = tf.compat.v1.Variable(tf.random_normal([2]),name='weights')
print(변수) #<tf.Variable 'weights:0' shape=(2,) dtype=float32_ref>

# 변수를 초기화하지 않으면 해당 변수가 메모리에 할당되지만 무작위로 초기화된다.
#

# 초기화 첫번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(변수)
print('aaa : ', aaa)    #aaa :  [-1.5080816   0.26086742]
sess.close()

# 초기화 두번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session= sess)  #텐서플로우 데이터형인 "변수"를 파이썬에서 볼 수 있게 바꿔줌
print('bbb : ', bbb)    #bbb :  [-1.5080816   0.26086742]
sess.close()

# 초기화 세번째
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()
print('ccc : ', ccc)    #ccc :  [-1.5080816   0.26086742]
sess.close()