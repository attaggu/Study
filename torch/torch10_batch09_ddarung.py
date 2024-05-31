

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

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
train_csv = train_csv.fillna(test_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)
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
from torch.utils.data import TensorDataset
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)
from torch.utils.data import DataLoader
train_loader = DataLoader(train_set, batch_size= 320, shuffle=True)
test_loader = DataLoader(test_set, batch_size=320, shuffle=False)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim,32)
        self.linear2 = nn.Linear(32,16)
        self.linear3 = nn.Linear(16,8)
        self.linear4 = nn.Linear(8,8)
        self.linear5 = nn.Linear(8,output_dim)
        
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        return x
model = Model(9,1).to(DEVICE)



# 3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 모델 학습 함수
def train(model, criterion, optimizer, loader):
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()   # 배치당 그라디언트가 초기화됨 ( 배치를 주었기 때문 )
        hypothesis = model(x_batch) 
        loss = criterion(hypothesis,y_batch)
        
        #역전파
        loss.backward() #기울기(gradient)값 계산 (loss를 weight로 미분한 값)    역전파 시작
        optimizer.step() # 가중치(w) 수정(weight 갱신)  역전파 끝
        total_loss = total_loss + loss.item()
        
        
    return total_loss / len(loader) # 토탈로스 / 13(배치)
###########################################################
# 에포크 설정 및 학습
epochs = 88
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer,train_loader)
    if epoch % 100 == 0:
        print(f'epoch: {epoch}, loss: {loss:.4f}')  # verbose

def evaluate(model, criterion, loader):
    model.eval()  # 평가 모드 - 훈련때는 괜찮은데 평가때 dropout등 다 적용 안된채로 평가해야된다
    total_loss = 0
    y_true = []
    y_pred = []
    for x_batch, y_batch in loader:
        with torch.no_grad():
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    return total_loss / len(loader), np.array(y_true), np.array(y_pred)

loss2, y_true, y_pred = evaluate(model, criterion, test_loader)
print("최종 loss:", loss2)
###### 여기까지 model.evaluate ######

r2 = r2_score(y_true, y_pred)
print("R2:", r2)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print("RMSE:", rmse)