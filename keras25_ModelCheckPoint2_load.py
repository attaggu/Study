from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler





datasets = load_boston()
x = datasets.data   #.data= x
y = datasets.target #.target= y
import warnings
warnings.filterwarnings('ignore')   #ignore = warnings 안보게할때
print(x.shape)  #(506,13)
print(y.shape)  #(506,)
print(datasets.feature_names)
# ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.9,
                                                 random_state=37)



# 2. model

# model = Sequential()
# model.add(Dense(16, input_dim=13))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(1))

# from keras.callbacks import EarlyStopping,ModelCheckpoint


# es=EarlyStopping(monitor='val_loss',mode='min',
#                  patience=10,verbose=1,restore_best_weights=True,
#                  )
# mcp=ModelCheckpoint(monitor='val_loss',mode='auto',
#                     verbose=1,save_best_only=True,
#                     filepath='../_data/_save/MCP/keras25_MCP1.hdf5'   #경로저장
#                     )

# model.compile(loss='mae', optimizer='adam',metrics=['acc'])
# hist=model.fit(x_train,y_train, epochs=1000, batch_size=50,
#           validation_split=0.2,
#           callbacks=[es,mcp],
#           )
# print(hist.history['loss'])     #loss만 따로
# print(hist.history['val_loss'])     #val_loss만 따로

model=load_model('../_data/_save/MCP/keras25_MCP1.hdf5')
#가중치와 모델 전부 저장돼어 불러와진다

loss = model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
result=model.predict(x)
r2 = r2_score(y_test,y_predict)
print("loss:",loss)
print("R2_score:",r2)




# loss: [4.944716930389404, 0.0]
# R2_score: 0.4353930567717563
# PS C:\Study> 
