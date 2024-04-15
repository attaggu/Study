import tensorflow as tf

tf.compat.v1.enable_eager_execution()   # 즉시실행 켜 
#즉시 실행모드 킴 - T2
# tensorflow version :  1.14.0
# 즉시실행모드 :  True
# GPU NO

# tensorflow version :  2.9.0
# 즉시실행모드 :  True
# PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')

tf.compat.v1.disable_eager_execution()  # 즉시실행 꺼 
#즉시 실행모드 끔 - 그래프연산모드 - T1 코드 사용가능

# tensorflow version :  1.14.0
# 즉시실행모드 :  False
# GPU NO

# tensorflow version :  2.9.0
# 즉시실행모드 :  False
# PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')

print("tensorflow version : ", tf.__version__)
print("즉시실행모드 : ", tf.executing_eagerly())

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print(gpus[0])
    except RuntimeError as e:
        print(e)
else :
    print("GPU NO")
    

    