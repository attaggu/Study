import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split,KFold,cross_val_score

#1. Data
x, y = load_iris(return_X_y=True)
n_splits=5
# kfold = KFold(n_splits=n_splits,shuffle=True,random_state=28)
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=28)

#2. Model
model=SVC()

#3. Compile, Fit
scores = cross_val_score(model,x,y,cv=kfold)
#4. Evaluate, Predict

print("acc:",scores, "\ncv-acc:", round(np.mean(scores),4))
