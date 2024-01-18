from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE:",rmse)

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor ='val_loss',
                   mode ='min',
                   patience =10,    #10번 참는다
                   verbose =1
                   )   #earlystopping 지점을 볼수있다