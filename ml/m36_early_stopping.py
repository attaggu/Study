from xgboost import XGBClassifier,XGBRegressor

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler

x,y=load_diabetes(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=111,train_size=0.8)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {
    'n_estimators' : 1000,
    'learning_rate' : 0.1,
    'max_depth' : 6,
    'min_child_weight' : 10,
    # 'subsample' : [0,0.1,0.2,0.3,0.5,0.7,1],
    # 'colsample_bytree' : [0,0.1,0.2,0.3,0.5,0.7,1],
    # 'colsample_bylevel' : [0.2,0.3,0.5,0.7,1],
    # 'colsample_bynode' : [0,0.1,0.2,0.3,0.5] ,
    # 'reg_alpha' : [0,0.1,0.01,0.001,1,2,10] ,
    # 'reg_lambda' : [0,0.1,0.01,1,2],
}
# 2. model
model= XGBRegressor(random_state=111, **parameters)
# model = RandomizedSearchCV(xgb, parameters, cv=kFold,n_jobs=-3)

# 3. fit
model.set_params(early_stopping_rounds=2,
                random_state=11111,
                **parameters
                )

model.fit(x_train,y_train,
          eval_set=[(x_test,y_test)],
          verbose=True
          )
results = model.score(x_test,y_test)


print("사용 파라미터:",model.get_params())
print("최종 점수:", results)