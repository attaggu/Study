import numpy as np
import pandas as pd
import hyperopt as hp

print(hp.__version__)   # 0.2.7

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

search_space = {'x1' : hp.quniform('x1', -10,10,1),
                'x2' : hp.quniform('x2', -15,15,1)}
                     # hp.quniform(label, low,high,q) - low 부터 high 까지 균등 분포(q=1)

# hp.quniform(labael, low,high,q) : label로 지정된 입력 값 변수 검색 공간을 최소값 low에서 최대값 high까지 q의 간격을 가지고 설정

# hp.uniform(label, low,high) : 최소값 low에서 최대값 high까지 정규분포 형태의 검색 공간 설정

# hp.randint(label, upper) : 0부터 최대값 upper까지 random한 정수값으로 검색 공간 설정

# hp.loguniform(label, low,high) : exp(uniform(low,high))값을 변환하며, 변환값의 log변환 된 값은 정규분포 형태를 가지는 검색 공간 설정                     
# y에서 많이 함 - ( 1, 2, 10, 10000, 10000000)이럴때 앞에 log를 씌어줘서 틀어진 값을 완화

def objective_func (search_space):
    x1 = search_space['x1']
    x2 = search_space['x2']
    return_value = x1**2 -20*x2
    
    return return_value



trial_val = Trials()

best = fmin(
    fn = objective_func,
    space= search_space,
    algo= tpe.suggest,  # 알고리즘, 디폴트
    max_evals=20,
    trials= trial_val,
    rstate= np.random.default_rng(seed=10),      
)

print(best)

print(trial_val.results)    #max_evals 만큼 함


print(trial_val.vals)   #찾은 값들


import pandas as pd

target = [ i['loss'] for i in trial_val.results]

print(target)    
# [-216.0, -175.0, 129.0, 200.0, 240.0, -55.0, 209.0, -176.0, -11.0, -51.0, 136.0, -51.0, 164.0, 321.0, 49.0, -300.0, 160.0, -124.0, -11.0, 0.0]

df = pd.DataFrame({'target' : target,
                   'x1' : trial_val.vals['x1'],
                   'x2' : trial_val.vals['x2'],                   
                   })

print(df)