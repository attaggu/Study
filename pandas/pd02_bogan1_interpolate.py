import pandas as pd
import numpy as np
from datetime import datetime

dates=['2/16/2024',
       '2/17/2024',
       '2/18/2024',
       '2/19/2024',
       '2/20/2024',
       '2/21/2024']
dates=pd.to_datetime(dates)


print(dates)
# DatetimeIndex(['2024-02-16', '2024-02-17', '2024-02-18', '2024-02-19', '2024-02-20', '2024-02-21']

print("=====================================")
ts = pd.Series([2, np.nan, np.nan, 8, 10, np.nan],index=dates)

print(ts)
print("=====================================")

ts = ts.interpolate()   #interpolate = 보간하다
print(ts)
# 2024-02-16     2.0      =>      2024-02-16     2.0
# 2024-02-17     NaN      =>      2024-02-17     4.0
# 2024-02-18     NaN      =>      2024-02-18     6.0
# 2024-02-19     8.0      =>      2024-02-19     8.0
# 2024-02-20    10.0      =>      2024-02-20    10.0
# 2024-02-21     NaN      =>      2024-02-21    10.0

# 1. 행 또는 열 삭제
# 2.임의의 값
# 평균 : mean 
# 중위 : median
# 0 : fillna
# 앞에값 : ffill
# 뒤에값 : bfill
# 특정값 : 지정
# 3. 보간 : interpolate
# 4. 모델 : predict
# 5. 부스팅 계열 : 통상 결측치 이상치에 대해 자유롭다
