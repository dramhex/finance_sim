import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42) #reproductiblity

n_steps = 100
n_sims = 10

s0 = 100                                    #initial price
epsilon = np.random.randn(n_steps, n_sims)  #noise
sigma = 0.01                                #volatility
mu = 0.001                                  #drift
prices = 1 + mu + (epsilon * sigma)         #gross returns

multipliers = s0 * np.cumprod(prices, axis=0) #prices series, geometric brownian motion

plt.plot(multipliers)
plt.show()