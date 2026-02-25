import numpy as np
import matplotlib.pyplot as plt

class GBMSimulator:
    # simple GBM simulator
    def __init__(self, s0, mu, sigma, n_steps=100, n_sims=1,
                 stop_loss=0, take_profit=np.inf):
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.n_steps = n_steps
        self.n_sims = n_sims
        self.prices_series = None
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def run_simulation(self):
        # generate random shocks and build price matrix
        eps = np.random.randn(self.n_steps, self.n_sims)
        gross = 1 + self.mu + self.sigma * eps
        self.prices_series = self.s0 * np.cumprod(gross, axis=0)
    
    def apply_sl_tp(self, stop_loss=0, take_profit=np.inf):
        # apply stop-loss / take-profit to existing prices
        prices = self.prices_series.copy()
        mask_sl = prices < stop_loss
        mask_tp = prices >= take_profit
        mask = mask_sl | mask_tp

        first = np.argmax(mask, axis=0)
        exit_prices = prices[first, np.arange(self.n_sims)]
        closing = np.where(exit_prices <= stop_loss, stop_loss, take_profit)

        pers = np.maximum.accumulate(mask, axis=0)
        return np.where(pers, closing, prices)
    
    @property
    def expected_wealth(self):
        return self.prices_series[-1].mean()
    
    @property
    def std_dev(self):
        return self.prices_series[-1].std()

    @property
    def loss_probability(self):
        return (self.prices_series[-1] < self.s0).mean()
    
    @property
    def var(self):
        return self.s0 - np.percentile(self.prices_series[-1], 5)
        
    def plot_simulation(self):
        plt.plot(self.prices_series)
        plt.title(f"{self.n_sims} GBM paths")
        plt.xlabel("step")
        plt.ylabel("price")
        plt.show()

    def plot_result_distribution(self):
        plt.hist(self.prices_series[-1], bins=self.n_sims // 10)
        plt.title("Final price distribution")
        plt.show()