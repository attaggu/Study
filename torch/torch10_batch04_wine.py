

# dacon_wine
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)

# 1. 데이터 
path = "c://_data//dacon//wine//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

# Mapping 'type' column
train_csv['type'] = train_csv['type'].map({'white': 1, 'red': 0}).astype(int)
test_csv['type'] = test_csv['type'].map({'white': 1, 'red': 0}).astype(int)

# Check unique values in the target column to confirm the number of classes
num_classes = train_csv['quality'].nunique()
print(f'Number of classes: {num_classes}')

x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']
print(x.shape, y.shape)
yo = pd.get_dummies(y)
x_train, x_test, y_train, y_test = train_test_split(x, yo, shuffle=True, random_state=12, train_size=0.8)

# numpy 데이터기 때문에 torch.FloatTensor하기 전에 변환 / 사용하려면 .cpu 붙여야 함
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
# y_train = torch.LongTensor(y_train.to_numpy()).to(DEVICE)
# y_test = torch.LongTensor(y_test.to_numpy()).to(DEVICE)
y_train = torch.FloatTensor(y_train.to_numpy()).to(DEVICE)
y_test = torch.FloatTensor(y_test.to_numpy()).to(DEVICE)


print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

from torch.utils.data import TensorDataset
train_set = TensorDataset(x_train,y_train)
test_set = TensorDataset(x_test,y_test)
from torch.utils.data import DataLoader
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

print(x_train.shape[1]) #12
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):  # init - layer 구성 정의
        super(Model, self).__init__()   # init - 이 모델을 어떻게 구성할지
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32,16)
        self.linear4 = nn.Linear(16,8)
        self.linear5 = nn.Linear(8, output_dim)        
        self.relu = nn.ReLU()
        
        
        return
    
    # 순전파
    def forward(self, input_size):              # forwrd nn.Module에서 상속받은.. / 모델 구성
        x = self.linear1(input_size)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        return x


model = Model(12,7).to(DEVICE)


# 3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()  # criterion : 표준
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 모델 학습 함수
def train(model, criterion, optimizer, loader):
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        print(hypothesis,y_train)
        loss = criterion(hypothesis,y_batch) #예상값과 실제값 loss   predict, y 비교 
    
        loss.backward() #기울기(gradient)값 계산 (loss를 weight로 미분한 값)    역전파 시작
        optimizer.step() # 가중치(w) 수정(weight 갱신)  역전파 끝
        total_loss = total_loss + loss.item()
    return total_loss / len(loader)

epochs = 20
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epoch: {}, loss: {}'.format(epochs, loss))   #verbose


def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    for x_batch, y_batch in loader:
        with torch.no_grad():
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    return total_loss / len(loader), y_true, y_pred

# 모델 평가
loss2, y_true, y_pred = evaluate(model, criterion, test_loader)
print("최종 loss:", loss2)

# 예측값 추출 및 정확도 계산
y_pred = np.array(y_pred)
y_test = np.argmax(y_test.cpu().numpy(), axis=1)
print("==================================================")
print(y_pred, y_test)
score = accuracy_score(y_test, y_pred)
print('accuracy: {:.4f}'.format(score))