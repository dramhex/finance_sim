import numpy as np
import matplotlib.pyplot as plt

class GBMSimulator:
    def __init__(self, s0, mu, sigma, n_steps = 100, n_sims = 1, stop_loss = 0, take_profit = np.inf):
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.epsilon = None
        self.n_steps = n_steps
        self.n_sims = n_sims
        self.prices_series = None
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def run_simulation(self):
        self.epsilon = np.random.randn(self.n_steps, self.n_sims)
        gross_prices = 1 + self.mu + (self.sigma*self.epsilon)
        prices_series = self.s0 * np.cumprod(gross_prices, axis=0)

        if self.stop_loss > 0 or self.take_profit < np.inf: 
            mask_sl = prices_series < self.stop_loss
            mask_tp = prices_series >= self.take_profit
            mask_trigger = mask_tp | mask_sl
            # Get index of first impact
            first_exit = np.argmax(mask_trigger, axis=0)
            # Get price at this exact time
            exit_prices = prices_series[first_exit, np.arange(self.n_sims)]
            # We apply the correct closing prices (SL or TP)
            closing_prices = np.where(exit_prices <= self.stop_loss, self.stop_loss, self.take_profit)
            # Accumulate True on TP or SL mask
            persistent_mask = np.maximum.accumulate(mask_trigger, axis=0)
            # Apply closing prices
            prices_series = np.where(persistent_mask, closing_prices, prices_series)

        self.prices_series = prices_series
        return prices_series
    
    @property
    def expected_wealth(self):
        return self.prices_series[-1].mean()
    
    @property
    def std_dev(self):
        return self.prices_series[-1].std()
    
    @property
    def ratio(self):
        return self.expected_wealth/self.std_dev

    @property
    def loss_probability(self):
        loss = self.prices_series[-1] < self.s0
        return loss.mean()
    
    @property
    def var(self):
        return self.s0 - np.percentile(self.prices_series[-1], 5)
        
    def plot_simulation(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.prices_series)
        plt.title(f"Simulation de {self.n_sims} trajectoires (GBM)")
        plt.xlabel("Temps (Steps)")
        plt.ylabel("Prix ($S_t$)")
        plt.grid(True)
        plt.show()

    def plot_result_distribution(self):
        plt.figure(figsize=(10, 6))
        plt.hist(self.prices_series[-1], bins=self.n_sims//10)
        plt.title(f"Histogramme des prix finaux sur {self.n_sims} simulations")
        plt.xlabel("Prix ($S_{final}$)")
        plt.ylabel("Nombre d'occurences")
        plt.grid(True)
        plt.show()

    
