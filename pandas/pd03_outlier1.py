import numpy as np
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])










'''
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75]) 
    #25 - quartile_1 -1사분위
    #50 - q2 - 중위
    #75 - quartile_3 - 3사분위
    #percentile - 백분위수
    print("1사분위:", quartile_1)
    print("q2:", q2)
    print("3사분위:", quartile_3)
    iqr = quartile_3 - quartile_1
    #iqr 사분위수 범위 = q3 - q1 
    print("iqr:", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    #이상치 경계 - 이 경계를 벗어나는 데이터를 찾기 위해 np.where 함수를 사용함
    return np.where((data_out>upper_bound) | (data_out<lower_bound))    # | = or 
outliers_loc = outliers(aaa)

print("이상치의 위치:", outliers_loc)
import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()
'''
