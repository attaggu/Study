import tensorflow as tf

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

node3 = tf.add(node1, node2)        # 더하기
node4 = tf.subtract(node1, node2)   # 빼기
node5 = tf.multiply(node1, node2)   # 곱하기
node6 = tf.divide(node1, node2)     # 나누기

sess = tf.Session()

print(sess.run(node3))
print(sess.run(node4))
print(sess.run(node5))
print(sess.run(node6))
