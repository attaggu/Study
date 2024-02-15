# split 후 slcaer , pca

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
print(sk.__version__)   #1.1.3

datasets = load_iris()
x = datasets['data']
y = datasets.target


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=888,shuffle=True,stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



pca = PCA(n_components=3)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


model = RandomForestClassifier(random_state=888)
model.fit(x_train,y_train)
results = model.score(x_test,y_test)
print('model.score:', results)



# (150, 4) (150,)
# model.score: 1.0
# pca 적용
# (150, 2) (150,)
# model.score: 0.8666666666666667