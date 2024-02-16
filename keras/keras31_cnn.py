from tensorflow.python.keras.models import Sequential
# from tensorflow.keras.models import Sequential
# from keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D    #2d이미지


model = Sequential()
# model.add(Dense(10,input_shape=(3,)))   # 인풋은 (n,3)
model.add(Conv2D(15, (3,3), input_shape=(10,10,1)))       # (10,10,1) = 10*10그림 1=흑백 / (10,10,3) = 10*10그림 3=칼라 / (3,3) = 연산할 픽셀 컷팅사이즈 갯수 / 15 = 다음 레이어로 전달해주는 아웃풋값
model.add(Dense(5))
model.add(Dense(1))
