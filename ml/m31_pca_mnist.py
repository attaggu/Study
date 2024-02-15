from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

(x_train,_),(x_test,_) = mnist.load_data()  #pca에 y는 필요없음

print(x_train.shape , x_test.shape) #(60000, 28, 28) (10000, 28, 28)

x = np.concatenate([x_train, x_test], axis=0)

print(x.shape)  #(70000, 28, 28)

# scaler = StandardScaler()
x =x.reshape(-1,28*28)

max_len_components = x.shape[1]

pca=PCA(n_components=max_len_components)

x= pca.fit_transform(x)
evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)



print(np.argmax(evr_cumsum>= 0.95)+1)    #154번째부터
print(np.argmax(evr_cumsum>= 0.99)+1)    #331번째부터
print(np.argmax(evr_cumsum>= 0.999)+1)   #486번째부터
print(np.argmax(evr_cumsum>= 1.0)+1)     #713번째부터 1
#argmax는 0부터 시작 evr_cumsum은 1부터 시작


'''
over_95 = np.sum(evr_cumsum >= 0.95)
over_99 = np.sum(evr_cumsum >= 0.99)
over_999 = np.sum(evr_cumsum >= 0.999)
over_1 = np.sum(evr_cumsum >= 1.0)



print(over_95)
print(over_99)
print(over_999)
print(over_1)
    
'''
