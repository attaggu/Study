from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

text = "나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다."

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
#{'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
print(token.word_counts)
#OrderedDict([('나는', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)])



x=token.texts_to_sequences([text])    #수치화
print(x)    #[[4, 2, 2, 3, 3, 5, 6, 7, 1, 1, 1, 8]]


# OneHotEncoder
x1=np.array(x).reshape(-1,1)
print(x1)
ohe=OneHotEncoder(sparse=False)
x1_ohe=ohe.fit_transform(x1)
print(x1_ohe.shape) #(12, 8)
x1_ohe=x1_ohe.reshape(-1,12,8)
print(x1_ohe.shape) #(1, 12, 8)

# pd.get_dummies
x2=np.array(x).reshape(-1)
x3=pd.get_dummies(x2)
# x2=pd.get_dummies(np.array(x).reshape(-1))    #위랑 똑같다.
# print(x3.shape)
x4=x3.values.reshape(-1,12,8)
# print(x4.shape)   #(1, 12, 8)

# to_categorical
x5 = to_categorical(x)
# print(x5)
# [[[0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]
x6=x5[:,:,1:]
print(x6.shape) #(1, 12, 9)
# to_categorical 는 0부터 시작인데 데이터는 1부터 시작 - 슬라이싱 해줘야함
# print(x1.shape)


# train_csv['대출기간'] = train_csv['대출기간'].str.slice(start=0,stop=3).astype(int