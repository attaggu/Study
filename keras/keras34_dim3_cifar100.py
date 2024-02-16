# acc = 0.4 이상

from keras.datasets import cifar100
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical

# 1.  데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)
unique, count = np.unique(y_train, return_counts=True)
print(unique)
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
#  48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
#  72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
#  96 97 98 99]
print(count)
# [500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500
#  500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500
#  500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500
#  500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500
#  500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500
#  500 500 500 500 500 500 500 500 500 500]

print(x_train.shape[0]) # 50000
print(x_train.shape[1]) # 32
print(x_train.shape[2]) # 32

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)


# 2 모델
model = Sequential()
model.add(Dense(100, input_shape=(32*32*3,), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(30, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=1000, verbose=1, validation_split=0.2, callbacks=[es])

#저장
model.save("c:\_data\_save\cifar100\keras34_cnn6_0122_1.h5")  

# 4 평가, 예측
results = model.evaluate(x_test, y_test)
predict = np.argmax(model.predict(x_test), axis=1)
acc_score = accuracy_score(np.argmax(y_test, axis=1), predict)
print('loss : ', results[0])
print('acc : ', results[1])
print('acc : ', acc_score)
print(predict)

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(history.history['val_acc'], color = 'blue', label = 'val_acc', marker = '.')
plt.plot(history.history['val_loss'], color = 'red', label = 'val_loss', marker = '.')
plt.show()

# keras31_cnn6_0122_1.h5
# loss :  1.8344695568084717
# acc :  0.5218999981880188
# acc :  0.5219