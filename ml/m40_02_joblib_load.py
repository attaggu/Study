from xgboost import XGBClassifier,XGBRegressor

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import accuracy_score, f1_score,r2_score,roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import joblib

x,y=load_digits(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=111,train_size=0.8,stratify=y)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. model
path = "c://_data//_save//_joblib_test//"
model = joblib.load(path + 'm40_joblib1_save.dat')

results = model.score(x_test,y_test)
y_predict=model.predict(x_test)
print("",results)

# print('r2:',r2_score(y_test,y_predict))
# print("사용 파라미터:",model.get_params())
# print("최종 점수:", results)
# print('acc:', accuracy_score(y_test,y_predict))
# print('f1:', f1_score(y_test,y_predict))
# print('auc:', roc_auc_score(y_test,y_predict))

