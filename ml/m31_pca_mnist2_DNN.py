from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import time
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
(x_train,y_train),(x_test,y_test) = mnist.load_data()  #pca에 y는 필요없음

start_time=time.time()
x = np.concatenate([x_train, x_test], axis=0)
y= np.concatenate([y_train,y_test],axis=0)
print(x.shape)  #(70000, 28, 28)

x =x.reshape(-1,28*28)
# y =y.reshape(-1,1)

components=[154,331,486,713,-1]

for num in components:
    if num ==-1:
        pca = PCA()
    else:
        pca=PCA(n_components=num)
    xpca=pca.fit_transform(x)
    
    x_train,x_test,y_train,y_test = train_test_split(xpca,y,train_size=0.8)
    
    model = Sequential()
    model.add(Dense(10,input_shape=(xpca.shape[1],)))
    model.add(Dense(10))
    model.add(Dense(10))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
    
    # 모델 훈련
    model.fit(x_train,y_train,epochs=33, verbose=1)
    
    # 모델 평가
    result=model.evaluate(x_test,y_test, verbose=1)
    print(f"Number of components: {num}")
    print("loss:" , result[0])
    print("acc:" , result[1])
end_time=time.time()
print("time:",round(end_time-start_time,2), "초")
'''

x= pca.fit_transform(x)
print(x.shape,y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8)
model = Sequential()

model.add(Dense(10,input_shape=(154,)))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])

model.fit(x_train,y_train,epochs=10,batch_size=100)
end_time=time.time()

result=model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
# acc=accuracy_score(y_test,y_predict)
print("loss:" , result[0])
print("acc:" , result[1])

print("time:",round(end_time-start_time,2), "초")
'''
