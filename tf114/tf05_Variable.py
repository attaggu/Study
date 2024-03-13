import tensorflow as tf
sess = tf.compat.v1.Session()

a = tf.Variable([2], dtype=tf.float32)
b = tf.Variable([2], dtype=tf.float32)

init = tf.compat.v1.global_variables_initializer()  # 변수 초기화 ( 변수를 적용한 다음 )
sess.run(init)

print(sess.run(a + b))
