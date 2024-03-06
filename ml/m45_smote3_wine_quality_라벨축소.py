import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from imblearn.over_sampling import SMOTE

path = "c://_data//dacon//wine//"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv",index_col=0)

submission_csv=pd.read_csv(path + "sample_submission.csv")

train_csv['type']=train_csv['type'].map({'white':1,'red':0}).astype(int)
test_csv['type']=test_csv['type'].map({'white':1,'red':0}).astype(int)


x=train_csv.drop(['quality'],axis=1)
y=train_csv['quality']-3
print(train_csv['quality'].value_counts().sort_index())
print(x.shape,y.shape)  #(5497, 12) (5497,)

label_counts = train_csv['quality'].value_counts().sort_index()

# 최소 그룹 크기를 설정합니다.
min_group_size = 800

# 그룹을 초기화합니다.
groups = []

# 현재 그룹을 초기화합니다.
current_group = []

# 각 라벨의 개수를 반복하면서 그룹을 형성합니다.
for label, count in label_counts.items():
    # 현재 라벨을 현재 그룹에 추가합니다.
    current_group.append(label)
    
    # 현재 그룹의 크기가 최소 그룹 크기보다 큰지 확인합니다.
    if sum([label_counts[group_label] for group_label in current_group]) > min_group_size:
        # 그룹을 최종 그룹에 추가하고 새로운 그룹을 시작합니다.
        groups.append(current_group[:-1])  # 마지막 라벨은 새 그룹에 포함시키지 않습니다.
        current_group = [label]

# 남은 그룹이 있다면 최종 그룹에 추가합니다.
if current_group:
    groups.append(current_group)

# 결과를 출력합니다.
print(groups)


x,_,y,_ = train_test_split(x,y,train_size=0.9,random_state=44,stratify=y)


print(x.shape,y.shape)  #(4947, 12) (4947,)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=44,stratify=y)


print(np.unique(y,return_counts=True))
print(pd.value_counts(y))


# print(np.unique(y_train,return_counts=True))


smote=SMOTE(random_state=44,k_neighbors=3)
x_train,y_train=smote.fit_resample(x_train,y_train)
print(np.unique(y_train,return_counts=True))


'''
print(y)
print("=============================================")
# x=x[30:]    #앞에서부터 30개 삭제
# y=y[30:]
# print(y)

# print(np.unique(y,return_counts=True))
x=x[:-35]   #뒤에서부터 30개 삭제
y=y[:-35]
print(y)
print(np.unique(y,return_counts=True))
#(array([0, 1, 2]), array([59, 71, 18], dtype=int64))

print(np.unique(y,return_counts=True))



#model
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,
                                               shuffle=True,
                                               random_state=14,
stratify=y)

print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)




'''