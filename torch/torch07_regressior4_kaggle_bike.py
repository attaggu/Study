from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)

# 1. 데이터
path = "c:\\_data\\dacon\\ddarung\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "submission.csv")

# 결측치 처리
train_csv = train_csv.dropna()
test_csv = test_csv.fillna(test_csv.mean())

# x = train_csv.drop(['count','casual','registered'],axis=1)
x = train_csv.drop(['count'],axis=1)

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=121)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, y_train.shape)  # (1167, 9) (1167,)
print(x_test.shape, y_test.shape)    # (292, 9) (292,)
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train.to_numpy()).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test.to_numpy()).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape)
# torch.Size([1167, 9]) torch.Size([1167, 1])
print(x_test.shape, y_test.shape)
# torch.Size([292, 9]) torch.Size([292, 1])
# 2. 모델구성
model = nn.Sequential(
    nn.Linear(9, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.08)

# 모델 학습 함수
def train(model, criterion, optimizer, x_train, y_train):
    model.train()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion( hypothesis,y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

# 에포크 설정 및 학습
epochs = 888
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    if epoch % 100 == 0:
        print(f'epoch: {epoch}, loss: {loss:.4f}')  # verbose

# 평가 함수
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    with torch.no_grad():
        y_predict = model(x_test)
        loss = criterion(y_test, y_predict)
    return loss.item(), y_predict

# 모델 평가
loss2, y_pred = evaluate(model, criterion, x_test, y_test)
print("최종 loss:", loss2)

# RMSE 계산
rmse = np.sqrt(loss2)
print("RMSE:", rmse)

# R2 스코어 계산
y_test_np = y_test.cpu().numpy()
y_pred_np = y_pred.cpu().numpy()
r2 = r2_score(y_test_np, y_pred_np)
print("R2:", r2)
