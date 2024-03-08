import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor,MultiOutputClassifier

# 1. Data

x,y=load_linnerud(return_X_y=True)

print(x.shape)  #(20, 3)
print(y.shape)  #(20, 3)

# 최종값 = x : [2. 110. 43.] y : [138. 33. 68.]

# 2. Model
# model = XGBRegressor()     #error
# model.fit(x,y)
# y_predict = model.predict(x)
# print(model.__class__.__name__, ' score :',
#       round(mean_absolute_error(y,y_predict),4))
# print(model.predict([[2, 110, 43]]))
# XGBRegressor  score : 0.0008
# [[138.0005    33.002136  67.99897 ]] 

# model = Ridge()
# model.fit(x,y)
# y_predict = model.predict(x)
# print(model.__class__.__name__, ' score :',
#       round(mean_absolute_error(y,y_predict),4))
# print(model.predict([[2, 110, 43]]))
# Ridge  score : 7.4569
# [[187.32842123  37.0873515   55.40215097]]




# model = LGBMRegressor()   #error - 컬럼이 여러개여서
# model.fit(x,y)
# y_predict = model.predict(x)
# print(model.__class__.__name__, ' score :',
#       round(mean_absolute_error(y,y_predict),4))
# print(model.predict([[2, 110, 43]]))
# Currently only multi-regression, multilabel and survival objectives work with multidimensional target


# model = MultiOutputRegressor(LGBMRegressor())   #error - 컬럼이 여러개여서 - MultiOutput으로 랩핑해줌
# model = MultiOutputRegressor(CatBoostRegressor(verbose=0))
model = CatBoostRegressor(loss_function='MultiRMSE',verbose=0)  
#CatBosst는 이 방법도 가능 - MultiRMSE로 훈련을 하고 위에 지정한 mae로 loss 구함
model.fit(x,y)
y_predict = model.predict(x)
print(model.__class__.__name__, ' score :',
      round(mean_absolute_error(y,y_predict),4))
print(model.predict([[2, 110, 43]]))