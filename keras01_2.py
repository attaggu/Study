# 맨 앞에서 컨트롤 / = 전체 주석변경

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2828)

loss = model.evaluate(x, y)
print("loss : ", loss)

result = model.predict([1,2,3,4,5,6,7])
print("7의 예측값 :", result)

# loss :  0.33687400817871094
# 1/1 [==============================] - 0s 56ms/step
# 7의 예측값 : [[6.9804883]]

# loss :  0.3467874228954315
# 1/1 [==============================] - 0s 32ms/step
# 7의 예측값 : [[7.0384083]]

# loss :  0.3406335413455963
# 1/1 [==============================] - 0s 49ms/step
# 7의 예측값 : [[0.9196938]
#  [1.9337436]
#  [2.9477935]
#  [3.9618433]
#  [4.9758925]
#  [5.9899426]
#  [7.0039926]]
# 에포 : 2828

