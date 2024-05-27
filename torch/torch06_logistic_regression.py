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
    nn.Linear(30, 64),
    nn.SELU(),
    nn.Linear(64, 32),
    nn.SELU(),
    nn.Linear(32, 16),
    nn.SELU(),
    nn.Linear(16, 8),
    nn.Linear(8, 1),
    nn.Sigmoid()
).to(DEVICE)
#############################


#3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam')
criterion = nn.BCELoss()                #criterion : 표준
# binary cross entropy
optimizer = optim.Adam(model.parameters(), lr = 0.01)
# optimizer = optim.SGD(model.parameters(), lr = 0.01)

# model.fit(x,y, epochs = 100, batch_size=1)
def train(model, criterion, optimizer, x_train, y_train):
    # model.train()   
    # 훈련모드 default - 훈련때는 괜찮은데 평가때 dropout등 다 적용 안된채로 평가해야된다
    # 그래서 평가때는 사용할수 없음
    optimizer.zero_grad()
    # w = w - lr * (loss를 weight로 미분한 값)
    hypothesis = model(x_train) #예상치 값 (순전파)   y_predict
    loss = criterion(hypothesis,y_train) #예상값과 실제값 loss   predict, y 비교 
    
    #역전파
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
        loss2 = criterion(y_test, y_predict)
    return loss2.item()
loss2 = evaluate(model,criterion,x_test,y_test)
# y_pred = model(x_test)
# 결과 device='cuda:0', grad_fn=<SigmoidBackward0>)

y_pred = np.round(model(x_test).cpu().detach().numpy())
print(y_pred)

score = accuracy_score(y_test.cpu().numpy(), y_pred)
print('accuracy:{:.4f}'.format(score))

    
    
