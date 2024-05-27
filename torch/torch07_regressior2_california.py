from sklearn.metrics import accuracy_score, mean_squared_error,r2_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)
# gpu 연산을 하려면 to.(DEVICE) 를 줘서 cuda로 돌리겠다고 정의

#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


print(x.shape,y.shape)  #(20640, 8) (20640,)
x_train,x_test,y_train,y_test = train_test_split(x,y)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape,y_train.shape)  #(15480, 8) (15480,)
print(x_test.shape,y_test.shape)    #(5160, 8) (5160,)


x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
print(x_train.shape,y_train.shape)
# torch.Size([15480, 8]) torch.Size([15480, 1])
print(x_test.shape,y_test.shape)   
# torch.Size([5160, 8]) torch.Size([5160, 1])



# print(x.shape, y.shape)     torch.Size([3, 1]) torch.Size([3, 1])
# print(x, y) #tensor([1., 2., 3.]) tensor([1., 2., 3.])
# y도 unsqueeze를 줘야함 - 쉐잎이 다르면 n빵 쳐서 계산하게됨

#2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))
# model = nn.Linear(1, 1).to(DEVICE) #input, output 케라스랑 반대
#############################
model = nn.Sequential(
    nn.Linear(8, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 2),
    nn.Linear(2, 1),
).to(DEVICE)
#############################


#3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam')
criterion = nn.MSELoss()                #criterion : 표준

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
    loss = criterion(hypothesis) #예상값과 실제값 loss   predict, y 비교 
    
    #역전파
    loss.backward() #기울기(gradient)값 계산 (loss를 weight로 미분한 값)    역전파 시작
    optimizer.step() # 가중치(w) 수정(weight 갱신)  역전파 끝
    return loss.item() #item 하면 numpy 데이터로 나옴
    
    
epochs = 888
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    if epoch % 100 == 0:
        print(f'epoch: {epoch}, loss: {loss:.4f}') 
    
#4. 평가, 예측
# loss = model.evaluate(x,y)
def evaluate(model, criterion, x_test, y_test ):
    model.eval()    # 평가 모드 - 훈련때는 괜찮은데 평가때 dropout등 다 적용 안된채로 평가해야된다
    with torch.no_grad():
        y_predict = model(x_test)
        loss2 = criterion(y_test, y_predict)
    return loss2.item(),y_predict # 평가 loss

loss2,y_pred = evaluate(model, criterion, x_test, y_test)
print("최종 loss :", loss2)
###### 여기까지 model.evaluate ######

y_test_np = y_test.cpu().numpy()
y_pred_np = y_pred.cpu().numpy()
r2 = r2_score(y_test_np, y_pred_np)
print("R2:", r2)

rmse = np.sqrt(loss2)
print("RMSE:", rmse)