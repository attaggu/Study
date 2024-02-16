import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input,Dense,Concatenate,concatenate



x1_data= np.array([range(100),range(301,401)]).T  
x2_data= np.array([range(101,201),range(411,511),range(150,250)]).T   
x3_data= np.array([range(100),range(301,401),range(77,177),range(33,133)]).T
print(x1_data.shape,x2_data.shape,x3_data.shape)    #(100, 2) (100, 3) (100, 4)

y1=np.array(range(3001,3101))   
y2=np.array(range(13001,13101))   
print(y1.shape,y2.shape)    #(100,) (100,)
x1_trian,x1_test,x2_train,x2_test,y1_train,y1_test,y2_train,y2_test = train_test_split(x1_data,x2_data,y1,y2,train_size=0.7,random_state=1)

input1=Input(shape=(2,))
dense1=Dense(10,activation='relu',name='bit1')(input1)
dense2=Dense(10,activation='relu',name='bit2')(dense1)
dense3=Dense(10,activation='relu',name='bit3')(dense2)
dense4=Dense(10,activation='relu',name='bit4')(dense3)
dense5=Dense(10,activation='relu',name='bit5')(dense4)
output1=Dense(10,activation='relu',name='bit6')(dense5)
# model1=Model(inputs=input1,outputs=output1)
# model1.summary()

input11=Input(shape=(3,))
dense11=Dense(100,activation='relu',name='bit11')(input11)
dense12=Dense(100,activation='relu',name='bit12')(dense11)
dense13=Dense(100,activation='relu',name='bit13')(dense12)
dense14=Dense(100,activation='relu',name='bit14')(dense13)
dense15=Dense(100,activation='relu',name='bit15')(dense14)
output11=Dense(5,activation='relu',name='bit16')(dense15)
# model2=Model(inputs=input11,outputs=output11)
# model2.summary()

#concatenate - model 두개를 엮음(list로)
# merge1=concatenate([output1,output11],name='mg1') 아래랑 똑같다-함수 클래스 차이
merge1=Concatenate(name='mg1')([output1,output11])

merge2=Dense(47,name='mg2')(merge1)
merge3=Dense(311,name='mg3')(merge2)
last_output1 = Dense(1,name='last')(merge3)

merge11=concatenate([output1,output11],name='mg11')
merge12=Dense(47,name='mg12')(merge11)
merge13=Dense(311,name='mg13')(merge12)
last_output11 = Dense(1,name='last1')(merge13)


model = Model(inputs=[input1,input11], outputs=[last_output1,last_output11])



model.compile(loss='mse',optimizer='adam')
model.fit([x1_trian,x2_train],[y1_train,y2_train],epochs=150)

result=model.evaluate([x1_test,x2_test],[y1_test,y2_test])
y_predict=model.predict([x1_test,x2_test])

print("??:",result)
print(y_predict)
print(np.array(y_predict).shape)
