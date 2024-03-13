import tensorflow as tf
print(tf.__version__)

print("tensorflow hello world")


hello = tf.constant('hello world')
print(hello)    # Session이고 hello 그래프만 불러온상태 

sess = tf.Session()

print(sess.run(hello))  # sess로 변수를 정의하고 run으로 출력

# b'hello world' - b = binary
