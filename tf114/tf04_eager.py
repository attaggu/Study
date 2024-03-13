import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())   # False - 디폴트
#즉시 실행모드 - tensorflow 1의 그래프 형태의 구성 없이 자연스러운 파이썬 문법으로 실행시킴

# tf.compat.v1.disable_eager_execution()  # 즉시실행모드 끔 - tensorflow 1.0 문법 - 디폴트
tf.compat.v1.enable_eager_execution()   # 즉시실행모드 켬 - tensorflow 2.0 문법 

print(tf.executing_eagerly())   # True

hello = tf.constant("Hello World")

sess = tf.compat.v1.Session()
print(sess.run(hello))


# 가상환경  즉시실행모드    사용가능
# 1.14.0 = disable(디폴트)   가능
# 1.14.0 = enable           에러
# 2.9.0 = disable           가능    tensorflow 2 환경에서 tensorflow 1 코드를 쓸일이 있을경우 사용
# 2.9.0 = enable(디폴트)     에러   



