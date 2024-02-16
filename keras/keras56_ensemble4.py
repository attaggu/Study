import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input,Dense,Concatenate,concatenate



x1_data= np.array([range(100),range(301,401)]).T  

y1=np.array(range(3001,3101))   
y2=np.array(range(13001,13101))   
print(y1.shape,y2.shape)    #(100,) (100,)
x1_trian,x1_test,y1_train,y1_test,y2_train,y2_test = train_test_split(x1_data,y1,y2,train_size=0.7,random_state=1)

input1=Input(shape=(2,))
dense1=Dense(10,activation='relu',name='bit1')(input1)
dense2=Dense(10,activation='relu',name='bit2')(dense1)
dense3=Dense(10,activation='relu',name='bit3')(dense2)
dense4=Dense(10,activation='relu',name='bit4')(dense3)
dense5=Dense(10,activation='relu',name='bit5')(dense4)
output1=Dense(10,activation='relu',name='bit6')(dense5)
# model=Model(inputs=input1,outputs=output1)
# model1=Model(inputs=input1,outputs=output1)

'''
# concatenate - model 두개를 엮음(list로)
merge1=concatenate([output1],name='mg1')
merge2=Dense(47,name='mg2')(merge1)
merge3=Dense(311,name='mg3')(merge2)
last_output = Dense(2,name='last')(merge3)
model = Model(inputs=[input1], outputs=last_output)
model.summary()

model.compile(loss='mse',optimizer='adam')
model.fit([x1_trian],[y1_train,y2_train],epochs=150)

result=model.evaluate([x1_test],[y1_test,y2_test])
y_predict=model.predict([x1_test])
# y_predict=model.predict()

'''
# merge1=concatenate([output1],name='mg1')
merge1=output1
merge2=Dense(47,name='mg2')(merge1)
merge3=Dense(311,name='mg3')(merge2)
last_output = Dense(1,name='last')(merge3)

merge11=output1
merge12=Dense(47,name='mg12')(merge11)
merge13=Dense(311,name='mg13')(merge12)
last_output1 = Dense(1,name='last1')(merge13)

model = Model(inputs=input1, outputs=[last_output,last_output1])
model.summary()




model.compile(loss='mse',optimizer='adam')
model.fit(x1_trian,[y1_train,y2_train],epochs=150)

result=model.evaluate(x1_test,[y1_test,y2_test])
y_predict=model.predict(x1_test)
# y_predict=model.predict()
print("??:",result)
print(y_predict)

