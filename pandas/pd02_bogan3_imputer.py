import pandas as pd
import numpy as np


data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                    [2, 4, np.nan, 8, np.nan],
                    [2, 4, 6, 8, 10],
                    [np.nan, 4, np.nan, 8, np.nan]])
print(data)
#      0    1    2  3     4
# 0  2.0  NaN  6.0  8  10.0
# 1  2.0  4.0  NaN  8   NaN
# 2  2.0  4.0  6.0  8  10.0
# 3  NaN  4.0  NaN  8   NaN
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = SimpleImputer()
# 결측치를 전부 평균(디폴트)으로 채워줌

data2 = imputer.fit_transform(data)
print(data2)

imputer = SimpleImputer(strategy='mean')    #평균
data3 = imputer.fit_transform(data)
print(data3)

imputer = SimpleImputer(strategy='median')  #중위
data4 = imputer.fit_transform(data)
print(data4)

imputer = SimpleImputer(strategy='most_frequent')   #가장 자주 나오는값
data5 = imputer.fit_transform(data)
print(data5)

imputer = SimpleImputer(strategy='constant')    #고정된 숫자 - 상수 / 0 
data6 = imputer.fit_transform(data)
print(data6)

imputer = SimpleImputer(strategy='constant',
                        fill_value=777) #조건에 따라 다른 숫자를 넣을수 있음
data7 = imputer.fit_transform(data)
print(data7)

imputer = KNNImputer()  #KNN 알고리즘
data8 = imputer.fit_transform(data)
print(data8)

imputer = IterativeImputer()    #선형회귀 알고리즘
data9= imputer.fit_transform(data)
print(data9)

from impyute.imputation.cs import mice
aaa= mice(data.values,seed=777,n=10)

print(aaa)