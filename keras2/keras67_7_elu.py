import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)


def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

y = elu(x)
plt.plot(x,y)
plt.grid()
plt.show()

