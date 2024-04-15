import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, Input


# 1. Data

(x_train,y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

# 2. Model

def build_model(drop=0.5, optimizer= 'adam', activation='relu',
                node1=64, node2=32, node3=16, lr=0.001):
    inputs = Input(shape=(28,28,1), name='inputs')
    x = Conv2D(filters=1, kernel_size=(3, 3), activation=activation)(inputs)  
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(node1, activation=activation, name = 'hidden1')(x)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='sparse_categorical_crossentropy')
    return model

def create_hyperparameter():
    batchs = [100,200,300,400,500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [ 64, 32, 16, 8]
    node2 = [64, 32, 16, 8]
    node3 = [64, 32, 16, 8]
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
keras_model = KerasClassifier(build_fn=build_model, verbose=1)

model = RandomizedSearchCV(keras_model, hyperparameters, cv=2,n_iter=1,
                           n_jobs=16, verbose=1)
import time
start_time = time.time()
model.fit(x_train, y_train, epochs=3)
end_time = time.time()

print(" time : ", round(end_time-start_time,2))
print('model.best_params_ :', model.best_params_)
print('model.best_estimator_ :', model.best_estimator_)
print('model.best_score_ :', model.best_score_)
print('model.score :', model.score(x_test,y_test))

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('acc_score :', accuracy_score(y_test,y_predict))

