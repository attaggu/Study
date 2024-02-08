import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split,KFold,cross_val_score
import pandas as pd
#1. Data
datasets = load_iris()

df=pd.DataFrame(datasets.data,columns=datasets.feature_names)
print(df)   #[150 rows x 4 columns] 
# x, y = load_iris(return_X_y=True)

n_splits=4
kfold = KFold(n_splits=n_splits,shuffle=True,random_state=28)
# kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=28)
for train_index, val_index in kfold.split(df):
    print("=========================")
    print(train_index, "\n", val_index)
    print("훈련데이터 개수",len(train_index),
          "검증데이터 개수",len(val_index))

'''
#2. Model
model=SVC()

#3. Compile, Fit
scores = cross_val_score(model,x,y,cv=kfold)
#4. Evaluate, Predict

print("acc:",scores, "\ncv-acc:", round(np.mean(scores),4))
'''
