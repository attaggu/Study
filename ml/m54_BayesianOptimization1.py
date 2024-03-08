param_bounds = {'x1' : (-1,5),
                'x2' : (0, 4),
                # 'x3' : (4, 2)
                }


def y_function(x1, x2):
    return -x1 **2 - (x2 -2) **2 +10


from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f = y_function, # y_function의 최대값
    pbounds = param_bounds, #파라미터 범위
    random_state= 123,
)

optimizer.maximize(
    init_points=5,n_iter=20,    # 총 25번 훈련

                   )
print(optimizer.max)