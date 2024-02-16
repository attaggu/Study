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

#결측치 확인
print(data.isnull())
#       x1     x2     x3     x4
# 0  False  False  False   True
# 1   True  False  False  False
# 2  False   True  False   True
# 3  False  False  False  False
# 4  False   True  False   True
# True - 결측치
print(data.isnull().sum())
# x1    1
# x2    2
# x3    0
# x4    3
print(data.info())
# RangeIndex: 5 entries, 0 to 4
# Data columns (total 4 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   x1      4 non-null      float64
#  1   x2      3 non-null      float64
#  2   x3      5 non-null      float64
#  3   x4      2 non-null      float64
# 1. 결측치 삭제
print(data.dropna())    #디폴트 axis=0
print(data.dropna(axis=1))

# 2. 결측치 평균
means = data.mean()
print(means)
data2 = data.fillna(means)
print(data2)

# 3. 결측치 중위
medians = data.median()
print(medians)
data3 = data.fillna(medians)
print(data3)

# 4. 결측치 0 / 임의의 값
print(data.fillna(0))

# 5. 결측치 앞에값 ffill
# print(data.ffill())
data5=data.ffill()
print(data5)

# 6. 결측치 뒤에값 bfill
# print(data.bfill())
data6=data.bfill()
print(data6)

# 특정 컬럼만
means = data['x1'].mean()
print(means)    #6.5

ff=data['x2'].ffill()

medians = data['x4'].median()
print(medians)  #6.0

data['x1'] = data['x1'].fillna(means)
data['x2'] = data['x2'].fillna(ff)
data['x4'] = data['x4'].fillna(medians)
print(data)