import numpy as np
from sklearn.preprocessing import PolynomialFeatures    
# 가상 feature - y = wx+ b 를 제곱해준다 (불균형한 그래프일때)

x = np.arange(8).reshape(4,2)

print(x)
# [[0 1]    -> 0,0,1
#  [2 3]    -> 4,6,9
#  [4 5]    -> 16,20,25
#  [6 7]]   -> 36,42,49
# 제곱 - w*x제곱 2w*x b

pf = PolynomialFeatures(degree=2,include_bias=False)
# degree -> 차수(2= x제곱, 3=x세제곱)   - 3 부터는 너무 커져서 잘 사용안함
# include=True -> x의 0제곱이 맨앞 컬럼에 들어감 - 전부 1이라 데이터활용에서 쓸모 없음 
x_pf = pf.fit_transform(x)

print(x_pf)
# [[ 0.  1.  0.  0.  1.]
#  [ 2.  3.  4.  6.  9.]
#  [ 4.  5. 16. 20. 25.]
#  [ 6.  7. 36. 42. 49.]]

print("===========================")

x = np.arange(12).reshape(4,3)
print(x)

pf = PolynomialFeatures(degree=2,include_bias=False)

x_pf = pf.fit_transform(x)
print(x_pf)