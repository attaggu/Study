from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.utils import pad_sequences
# from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,Reshape,Conv1D,LSTM,Embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping,ModelCheckpoint

# 1 Data
docs = [
    '너무 재미있다', ' 참 최고에요', '참 잘만든 영화에요', '추천하고 싶은 영화입니다',
    '한 번 더 보고 싶어요', '흠', '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밌어요', '배고프다', '볼만해요', '눕고싶다'
]

x_predict = ['나는 진짜 너무 재미없다 배고프다 졸리다 지루해요']


labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token=Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
#{'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7,
#'싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, 
#'흠': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, 
#'재미없어요': 21, '재미없다': 22, '재밌어요': 23, '배고프다': 24, '볼만해요': 25, '눕고싶다': 26}
#26 => 단어사전의 개수

token.fit_on_texts(x_predict)
print(token.word_index)
x_predict=token.texts_to_sequences(x_predict)

x = token.texts_to_sequences(docs)  #단어를 수치화
print(type(x))  #<class 'list'> - 리스트임
print(x)    
#[[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15],
#[16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24], [25], [26]]
#큰 데이터에 사이즈를 맞춤 - padding 처럼 0을 넣어 맞춤 ( 왼쪽에 넣는게 좋음 - 왼쪽부터 오른쪽 000262 순서로 학습을 하기 때문)
#너무 큰 문장은 일정 사이즈로 자름
# x=np.array(x)   #문장 개수가 달라서 안됨 - 사이즈를 맞춰야함

pad_x=pad_sequences(x,
                    padding='pre', #0을 채워넣음 pre -앞에 , post - 뒤에
                    # truncating='post',  #데이터가 넘칠때 pre - 앞에서 자름 , post - 뒤에서 자름
                    maxlen=5    #데이터 길이를 지정
                    )

pad_x_predict=pad_sequences(x_predict,
                            padding='pre',
                            maxlen=5)
y=labels
print(y.shape)  #(15,)

x=pad_x

# 2 Model
model=Sequential()
#=========================
model.add(Embedding(input_dim=30,output_dim=10,input_length=5))
# model.add(Embedding(27,10)) = 같다    input_dim , output_dim  적용   
#input_dim => 국어사전 개수+1
#output_dim => units
#input_length => (15,5)가 들어와서 행무시해서 5개(패딩 적용된 데이터 길이) = 연산에 의마 X, 자동으로 알아서 적용
#embedding 인풋의 shape = 2차원 / embedding 아웃풋의 shape = 3차원
#input_dim => 적으면 그수만큼 단어를 잘라서 훈련 / 많으면 임의 단어를 추가(증폭)해서 훈련 - but 증폭했지만 오히려 성능이 떨어질수 있음
#=========================

model.add(LSTM(10,input_shape=(5,1)))
model.add(Dense(4))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
model.add(Dense(6))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=32)

result=model.evaluate(x,y)
y_predict=np.around(model.predict(pad_x_predict))

print("loss:",result)
print(y_predict)
