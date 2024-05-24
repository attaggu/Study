import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#1. 데이터 
x = np.array([1,2,3])
y = np.array([1,2,3])

# x = torch.FloatTensor(x)
# y = torch.FloatTensor(y)
# print(x.shape, y.shape)   torch.Size([3]) torch.Size([3])

x = torch.FloatTensor(x).unsqueeze(1)
y = torch.FloatTensor(y).unsqueeze(1)
# print(x.shape, y.shape)     torch.Size([3, 1]) torch.Size([3, 1])
# print(x, y) #tensor([1., 2., 3.]) tensor([1., 2., 3.])
# y도 unsqueeze를 줘야함 - 쉐잎이 다르면 n빵 쳐서 계산하게됨

#2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))
model = nn.Linear(1, 1) #input, output 케라스랑 반대

#3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam')
criterion = nn.MSELoss()                #criterion : 표준

# optimizer = optim.Adam(model.parameters(), lr = 0.01)
optimizer = optim.SGD(model.parameters(), lr = 0.01)

# model.fit(x,y, epochs = 100, batch_size=1)
def train(model, criterion, optimizer, x, y):
    # model.train()   
    # 훈련모드 default - 훈련때는 괜찮은데 평가때 dropout등 다 적용 안된채로 평가해야된다
    # 그래서 평가때는 사용할수 없음
    optimizer.zero_grad()
    # w = w - lr * (loss를 weight로 미분한 값)
    hypothesis = model(x) #예상치 값 (순전파)   y_predict
    loss = criterion(y, hypothesis) #예상값과 실제값 loss   predict, y 비교 
    
    #역전파
    loss.backward() #기울기(gradient)값 계산 (loss를 weight로 미분한 값)    역전파 시작
    optimizer.step() # 가중치(w) 수정(weight 갱신)  역전파 끝
    return loss.item() #item 하면 numpy 데이터로 나옴
    
    
epochs = 2000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epochs, loss))   #verbose
    
#4. 평가, 예측
# loss = model.evaluate(x,y)
def evaluate(model, criterion, x, y ):
    model.eval()    # 평가 모드 - 훈련때는 괜찮은데 평가때 dropout등 다 적용 안된채로 평가해야된다
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item() # 평가 loss

loss2 = evaluate(model, criterion, x, y)
print("최종 loss :", loss2)
###### 여기까지 model.evaluate ######

# result = model.predict([4])
result = model(torch.Tensor([4]))
print('4의 예측값 :', result.item())

# 최종 loss : 8.146195682456892e-07
# 4의 예측값 : 4.0018110275268555
