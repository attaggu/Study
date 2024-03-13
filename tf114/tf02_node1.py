import tensorflow as tf

# 3 + 4 = ?
node1 = tf.constant(3.0, tf.float32)    # 3

node2 = tf.constant(4.0)    # 4 자동으로 위 float32

# node3 = node1 + node2
node3 = tf.add(node1, node2)

print(node3)    
# 그래프가 나옴 - Tensor("Add:0", shape=(), dtype=float32)

sess = tf.Session()
print(sess.run(node3))  # 7.0
