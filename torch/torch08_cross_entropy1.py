#digits

from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)
# gpu 연산을 하려면 to.(DEVICE) 를 줘서 cuda로 돌리겠다고 정의

#1. 데이터 

x, y = load_digits(return_X_y=True)

print(x.shape,y.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,random_state=12,train_size=0.8)
# numpy 데이터기 때문에 torch.FloatTensor하기전에 변환 / 사용하려면 .cpu 붙여야함
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape,x_test.shape)
# (1437, 64) (360, 64)
print(y_train.shape,y_test.shape)
# (1437,) (360,)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)

# gpu에서 사용할거라고 정의

# print(x.shape, y.shape)     torch.Size([3, 1]) torch.Size([3, 1])
# print(x, y) #tensor([1., 2., 3.]) tensor([1., 2., 3.])
# y도 unsqueeze를 줘야함 - 쉐잎이 다르면 n빵 쳐서 계산하게됨

#2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))
# model = nn.Linear(1, 1).to(DEVICE) #input, output 케라스랑 반대
#############################

model = nn.Sequential(
    nn.Linear(64, 64),
    nn.SELU(),
    nn.Linear(64, 32),
    nn.SELU(),
    nn.Linear(32, 16),
    nn.SELU(),
    nn.Linear(16, 8),
    nn.Linear(8, 10),
).to(DEVICE)
#############################


#3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam')
criterion = nn.CrossEntropyLoss()                #criterion : 표준

optimizer = optim.Adam(model.parameters(), lr = 0.001)
# optimizer = optim.SGD(model.parameters(), lr = 0.01)

# model.fit(x,y, epochs = 100, batch_size=1)
def train(model, criterion, optimizer, x_train, y_train):
    model.train()   
    optimizer.zero_grad()
    hypothesis = model(x_train) #예상치 값 (순전파)   y_predict
    loss = criterion(hypothesis,y_train) #예상값과 실제값 loss   predict, y 비교 
    
    loss.backward() #기울기(gradient)값 계산 (loss를 weight로 미분한 값)    역전파 시작
    optimizer.step() # 가중치(w) 수정(weight 갱신)  역전파 끝
    return loss.item() #item 하면 numpy 데이터로 나옴
    
epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss: {}'.format(epochs, loss))   #verbose


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
score = accuracy_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
print('accuracy: {:.4f}'.format(score))