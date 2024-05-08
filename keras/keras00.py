import tensorflow as tf
tf.__version__

## 방법 2-1 : 모든 사용 가능한 GPU List 보기
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# 방법 2-2
tf.config.list_physical_devices('GPU')