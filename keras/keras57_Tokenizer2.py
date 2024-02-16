from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

text = '나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'
text2 = '태규가 수업을 듣는다. 태규가 졸았다. 태규가 마구 마구 먹었다.'



token = Tokenizer()
token.fit_on_texts([text,text2])

print(token.word_index)
# {'마구': 1, '태규가': 2, '진짜': 3, '매우': 4, '먹었다': 5, '나는': 6, '맛있는': 7, '밥을': 8, '엄청': 9, '수업을': 10, '듣는다': 11, '졸았다': 12}
print(token.word_counts)
# OrderedDict([('나는', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 5), ('먹었다', 2), ('태규가', 3), ('수업을', 1), ('듣는다', 1), ('졸았다', 1)])

x=token.texts_to_sequences([text+text2])    #수치화
# print(x)    #[[6, 3, 3, 4, 4, 7, 8, 9, 1, 1, 1, 5, 2, 10, 11, 2, 12, 2, 1, 1, 5]]
# print(np.array(x).shape)    #(1,21)

# # 1. to_categorical
# x1 = to_categorical(x)
# print(x1)
# x2=x1[:,:,1:]   #맨앞 빈 o 데이터 삭제
# print(x2.shape) #(1, 21, 12)



# # 2. OneHotEncoder
# x3=np.array(x).reshape(-1,1)
# print(x3)
# ohe=OneHotEncoder(sparse=False)
# x3_ohe=ohe.fit_transform(x3)
# print(x3_ohe.shape) #(21, 12)
# x3_ohe=x3_ohe.reshape(-1,21,12)
# print(x3_ohe.shape) #(1, 21, 12)


# 3. pd.get_dummies
x4=np.array(x).reshape(-1)
print(x4.shape)
'''
x5=pd.get_dummies(x4)
print(x5.shape) #(21, 12)
# x5=pd.get_dummies(np.array(x).reshape(-1))    #위랑 똑같다.
x6=x5.values.reshape(-1,21,12)
print(x6.shape)   #(1, 21, 12)

'''


