from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)

# 1. 데이터 
path = "c://_data//dacon//dechul//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sub_csv = pd.read_csv(path + "sample_submission.csv")

# 데이터 전처리
train_csv.iloc[28730, 3] = 'OWN'
test_csv.iloc[34486, 7] = '기타'

le = LabelEncoder()
for col in ['주택소유상태', '대출목적', '근로기간']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])

train_csv['대출기간'] = train_csv['대출기간'].replace({' 36 months': 36, ' 60 months': 60}).astype(int)
test_csv['대출기간'] = test_csv['대출기간'].replace({' 36 months': 36, ' 60 months': 60}).astype(int)

x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']

# 원-핫 인코딩
ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y.values.reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=12, train_size=0.8)

# 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 텐서 변환
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).to(DEVICE)
y_test = torch.FloatTensor(y_test).to(DEVICE)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2. 모델구성
# model = nn.Sequential(
#     nn.Linear(x_train.shape[1], 64),
#     nn.SELU(),
#     nn.Linear(64, 32),
#     nn.SELU(),
#     nn.Linear(32, 16),
#     nn.SELU(),
#     nn.Linear(16, 8),
#     nn.Linear(8, y_train.shape[1])  # 다중 분류이므로 클래스의 수 만큼 출력 노드 설정
# ).to(DEVICE)

print(x_train.shape[1],y_train.shape[1])    #13 7


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):  # init - layer 구성 정의
        super(Model, self).__init__()   # init - 이 모델을 어떻게 구성할지
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32,16)
        self.linear4 = nn.Linear(16,8)
        self.linear5 = nn.Linear(8,7)
        self.sigmoid = nn.Sigmoid()
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


model = Model(13,7).to(DEVICE)



# 3. 컴파일, 훈련
criterion = nn.BCEWithLogitsLoss()  # 원-핫 인코딩된 타겟을 위한 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 모델 학습 함수
def train(model, criterion, optimizer, x_train, y_train):
    model.train()   
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 20
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss: {}'.format(epoch, loss))  # verbose

def evaluate(model, criterion, x_test, y_test):
    model.eval()
    with torch.no_grad():
        y_predict = model(x_test)
        loss = criterion(y_predict, y_test)
    return loss.item(), y_predict

# 모델 평가
loss2, y_pred = evaluate(model, criterion, x_test, y_test)
print("최종 loss:", loss2)

# 예측값 추출 및 정확도 계산
y_pred = torch.argmax(y_pred, dim=1)
y_test = torch.argmax(y_test, dim=1)
print("==================================================")
print(y_pred, y_test)
score = accuracy_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
print('accuracy: {:.4f}'.format(score))
