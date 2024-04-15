import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, Input
import pandas as pd

# 1. Data

x,y = fetch_california_housing(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,random_state=121,
                                                 train_size=0.8)
print(x_train.shape,x_test.shape)
# (16512, 8) (4128, 8)
print(y_train.shape, y_test.shape)
# (16512,) (4128,)
callbacks = [ EarlyStopping(monitor='loss', patience=1, restore_best_weights=True),
             ModelCheckpoint(filepath='best_model.h5', monitor='loss', save_best_only=True)]
# 2. Model

def build_model(drop=0.15, optimizer= 'adam', activation='relu',
                node1=128, node2=64, node3=32, lr=0.01):
    inputs = Input(shape=(8,), name='inputs')
    x = Dense(node1, activation=activation, name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1, name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['mse'],
                  loss='mse')
    return model

def create_hyperparameter():
    batchs = [100,200,300,400,500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    return {'batch_size' : batchs,
            'optimizer' : optimizers,
            'drop' : dropouts,
            'activation' : activations,
            'node1' : node1,
            'node2' : node2,
            'node3' : node3,
            }
hyperparameters = create_hyperparameter()
print(hyperparameters)

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
########## RandomizedSearchCv 에 optimizer, node 에 대한 parameter가 없기 때문에 적용가능한 
########## parameter을 줘야한다. - 케라스에서 사용할수있는 사이킥런 머신러닝으로 랩핑
keras_model = KerasRegressor(build_fn=build_model, verbose=1)

model = RandomizedSearchCV(keras_model, hyperparameters, cv=2,n_iter=13,
                           n_jobs=-1, verbose=1)
import time
start_time = time.time()
model.fit(x_train, y_train, epochs=15, callbacks=callbacks)
end_time = time.time()

print(" time : ", round(end_time-start_time,2))
print('model.best_params_ :', model.best_params_)
print('model.best_estimator_ :', model.best_estimator_)
print('model.best_score_ :', model.best_score_)
print('model.score :', model.score(x_test,y_test))

from sklearn.metrics import accuracy_score,r2_score,mean_squared_error
y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict)
print("r2 : ",r2)

