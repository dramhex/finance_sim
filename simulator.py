import numpy as np
import matplotlib.pyplot as plt

class GBMSimulator:
    def __init__(self, s0, mu, sigma, n_steps = 100, n_sims = 1):
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.epsilon = None
        self.n_steps = n_steps
        self.n_sims = n_sims
        
        self.prices_series = self.run_simulation()

        self.results = None

    def run_simulation(self):
        self.epsilon = np.random.randn(self.n_steps, self.n_sims)
        gross_prices = 1 + self.mu + (self.sigma*self.epsilon)
        prices_series = self.s0 * np.cumprod(gross_prices, axis=0)

        return prices_series
    
    def plot_simulation(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.prices_series)
        plt.title(f"Simulation de {self.n_sims} trajectoires (GBM)")
        plt.xlabel("Temps (Steps)")
        plt.ylabel("Prix ($S_t$)")
        plt.grid(True)
        plt.show()

    
