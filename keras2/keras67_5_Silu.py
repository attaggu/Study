import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5,5,0.1)

silu = lambda x: x * 1 / (1 + np.exp(-x))

y = silu(x)

plt.plot(x,y)
plt.grid()
plt.show()