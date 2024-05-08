import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)


y=leaky_relu(x)





plt.plot(x,y)
plt.grid()
plt.show()
