class 과일:
    def __init__(self, name):
        self.name = name

    def info(self):
        print(f"이 과일은 {self.name}이고")

class 사과(과일):
    def info(self):
        print(f"이 과일은 {self.name}이고 빨간색이다")

class 바나나(과일):
    def info(self):
        print(f"이 과일은 {self.name}이고 노란색이다")

class 체리(과일):
    def info(self):
        print(f"이 과일은 {self.name}이고 빨간색이고 달콤하다")

apple = 사과("사과")
banana = 바나나("바나나")
cherry = 체리("체리")

apple.info()
# 이 과일은 사과이고 빨간색이다

banana.info()
# 이 과일은 바나나이고 노란색이다

cherry.info()
# 이 과일은 체리이고 빨간색이고 달콤하다
