import numpy as np

a = np.array(range(1,11))
print(a)

size = 5
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size +1):
        # subset = dataset[i : (i + size)]
        # aaa.append(subset)
        aaa.append(dataset[i : (i+size)])
    return np.array(aaa)

bbb = split_x(a, size)

print(bbb)
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]
print(bbb.shape)    #(6,5)


x = bbb[:, :-1]
y = bbb[:, -1]
print(x,y)
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]] 
# [ 5  6  7  8  9 10]
print(x.shape,y.shape)  #(6, 4) (6,)
