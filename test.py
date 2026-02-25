import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

nombres = 300 + np.random.randn(30,10)

print(nombres)

plt.plot(nombres)
plt.show()
