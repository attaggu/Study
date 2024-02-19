from xgboost import XGBClassifier,XGBRegressor

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score,r2_score,mean_absolute_error

x,y=load_diabetes(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=111,train_size=0.8,)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {
    'n_estimators' : 2000,
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
model.set_params(early_stopping_rounds=1000,
                random_state=1111,
                **parameters
                )

model.fit(x_train,y_train,
          eval_set=[(x_train,y_train),(x_test,y_test)],
          verbose=10,
           #   eval_metric='rmse',   #디폴트 / 회귀
          eval_metric='mae',      #회귀
        #   eval_metric='rmsle',    #회귀       
        
          # eval_metric='logloss',  #이진분류디폴트 ACC
          # eval_metric='error',   #이진분류
        #   eval_metric='mlogloss',   #다중분류디폴트 ACC
        #   eval_metric='merror',   #다중분류
          # eval_metric='auc',      #이진분류 , 다중분류(이진에 더 좋음)
          )
results = model.score(x_test,y_test)
y_predict=model.predict(x_test)


print('mae:',mean_absolute_error(y_test,y_predict))
print("사용 파라미터:",model.get_params())
print("최종 점수:", results)
# print('acc:', accuracy_score(y_test,y_predict))
# print('f1:', f1_score(y_test,y_predict))
# print('auc:', roc_auc_score(y_test,y_predict))
print("=======================================================")

hist=model.evals_result()

print(hist)
train_mae = hist['validation_0']['mae']
val_mae = hist['validation_1']['mae']
import matplotlib.pyplot as plt
# plt.plot(hist['validation_0']['mae'], label='Training MAE',color='red')
# plt.plot(hist['validation_1']['mae'], label='Validation MAE',color='green')
plt.scatter(range(len(train_mae)), train_mae, label='Train MAE', marker='+')
plt.scatter(range(len(val_mae)), val_mae, label='Validation MAE', marker='+')
plt.xlabel('Number of Iterations')
plt.ylabel('MAE')
plt.title('Training Progress')
plt.legend()
plt.grid()
plt.show()


