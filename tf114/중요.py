

# Tensor 1 은 '그래프연산' 모드
# Tensor 2 는 '즉시실행' 모드

# tf.comopat.v1.enable_eager_execution()
# 즉시실행 모드 킴 -> Tensor 2 의 디폴트

# tf.compat.v1.disable_eager_execution()
# 즉시실행 모드 끔 -> Tensor 1 코드 쓸 수 있음
#                 -> 그래프연산 모드로 돌아감
                
# tf.executing_eagerly()
# True 면 즉시실행 모드 -> Tensor 2 코드만 써야함
# False면 그래프연산 모드 -> Tensor 1 코드를 쓸 수 있음