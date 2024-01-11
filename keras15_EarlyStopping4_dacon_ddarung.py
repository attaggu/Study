from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
path = "c:\\_data\\dacon\\ddarung\\"
train_csv = pd.read_csv(path + "train.csv",index_col=0)
test_csv = pd.read_csv(path + "test.csv",index_col=0)
submission_csv = pd.read_csv(path + "submission.csv")
train_csv = train_csv.dropna() 
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'],axis=1)
y = train_csv['count'] 
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,random_state=2086)

model=Sequential()
model.add(Dense(7, input_dim=9,activation='relu'))
model.add(Dense(7))
model.add(Dense(11,activation='relu'))
model.add(Dense(11,activation='relu'))
model.add(Dense(11))
model.add(Dense(11,activation='relu'))
model.add(Dense(11,activation='relu'))
model.add(Dense(11))
model.add(Dense(11,activation='relu'))
model.add(Dense(11,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
es=EarlyStopping(monitor='val_loss',mode='min',patience=277,verbose=1,restore_best_weights=True)
hist=model.fit(x_train,y_train, epochs=5000, batch_size=50,
          validation_split=0.3,callbacks=[es])
loss=model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
submission_csv['count']=y_submit 
submission_csv.to_csv(path + "submission_0106.csv", index=False)
y_predict=model.predict(x_test)

r2 = r2_score(y_test,y_predict)
print("loss:", loss)
print("R2_score:",r2)
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE:",rmse)
# print(hist.history['val_loss'])

# plt.rcParams['font.family']='Malgun Gothic'
# plt.rcParams['axes.unicode_minus']=False

# plt.figure(figsize=(100,100))
# plt.plot(hist.history['loss'],c='pink',label='loss',marker='.')
# plt.plot(hist.history['val_loss'],c='red',label='val_loss',marker='.')
# plt.legend(loc='upper right')
# plt.title('따릉이')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()
# plt.show()