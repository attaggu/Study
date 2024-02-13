from sklearn.model_selection import RandomizedSearchCV,train_test_split,GridSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold,cross_val_score,cross_val_predict
from sklearn.metrics import accuracy_score
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

x, y = load_digits(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,random_state=12,train_size=0.8)
model = make_pipeline(MinMaxScaler(),RandomForestClassifier())
start_time=time.time()
model.fit(x_train,y_train)
end_time=time.time()
y_predict=model.predict(x_test)
print("acc.score:", accuracy_score(y_test,y_predict))

print("time:",round(end_time-start_time,2),"s")
print("model.score:",model.score(x_test,y_test))

# random--
# acc.score: 0.9805555555555555
# best_acc.score: 0.9805555555555555
# time: 1.87 s
# model.score: 0.9805555555555555
# grid--
# acc.score: 0.9777777777777777
# best_acc.score: 0.9777777777777777
# time: 3.23 s
# model.score: 0.9777777777777777