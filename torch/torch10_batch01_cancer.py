# batch를 주려면 tensor x 데이터와 tensor y 데이터를 합친 후 배치를 줘야함 - 2번 작업
#이진분류
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)
# gpu 연산을 하려면 to.(DEVICE) 를 줘서 cuda로 돌리겠다고 정의

#1. 데이터 

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape,y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=950228,shuffle=True,stratify=y)
# numpy 데이터기 때문에 torch.FloatTensor하기전에 변환 / 사용하려면 .cpu 붙여야함
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.shape,x_test.shape)


# x와 y를 합치는 작업 - TensorDataset 형태로 

from torch.utils.data import TensorDataset
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

print(train_set)
# <torch.utils.data.dataset.TensorDataset object at 0x00000150155E4490>
print(len(train_set))

# 배치 넣어주는 작업 - DataLoader

from torch.utils.data import DataLoader
train_loader = DataLoader(train_set, batch_size= 32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
# Test는 섞을 필요가 없어 보통 shuffle=False

#############################################################################


# gpu에서 사용할거라고 정의

# print(x.shape, y.shape)     torch.Size([3, 1]) torch.Size([3, 1])
# print(x, y) #tensor([1., 2., 3.]) tensor([1., 2., 3.])
# y도 unsqueeze를 줘야함 - 쉐잎이 다르면 n빵 쳐서 계산하게됨

#2. 모델구성

# model = nn.Sequential(
#     nn.Linear(30, 64),
#     nn.SELU(),
#     nn.Linear(64, 32),
#     nn.SELU(),
#     nn.Linear(32, 16),
#     nn.SELU(),
#     nn.Linear(16, 8),
#     nn.Linear(8, 1),
#     nn.Sigmoid()
# ).to(DEVICE)
# class로 변환
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):  # init - layer 구성 정의
        super(Model, self).__init__()   # init - 이 모델을 어떻게 구성할지
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear2 = nn.Linear(32,32)
        self.linear3 = nn.Linear(32,16)
        self.linear4 = nn.Linear(16, output_dim)
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
        x = self.sigmoid(x)
        return x


model = Model(30,1).to(DEVICE)
#############################


#3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam')
criterion = nn.BCELoss()                #criterion : 표준
# binary cross entropy
optimizer = optim.Adam(model.parameters(), lr = 0.01)
# optimizer = optim.SGD(model.parameters(), lr = 0.01)

############################################################################
# model.fit(x,y, epochs = 100, batch_size=1)
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
######################################################################
epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epoch: {}, loss: {}'.format(epochs, loss))   #verbose


def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    for x_batch, y_batch in loader:
        with torch.no_grad():
            y_predict = model(x_batch)
            loss2 = criterion(y_predict, y_batch)
            total_loss = total_loss +loss2.item()
    return total_loss / len(loader)
loss2 = evaluate(model,criterion,test_loader)
print("최종 loss :", loss2)

y_pred = np.round(model(x_test).cpu().detach().numpy())
print(y_pred)

score = accuracy_score(y_test.cpu().numpy(), y_pred)
print('accuracy:{:.4f}'.format(score))

    
