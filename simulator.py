import numpy as np
import matplotlib.pyplot as plt

class GBMSimulator:
    """Geometric Brownian Motion price path simulator.

    Attributes:
        s0 (float): Initial asset price.
        mu (float): Drift term per step.
        sigma (float): Volatility per step.
        n_steps (int): Number of time steps per simulation.
        n_sims (int): Number of Monte Carlo trials.
        prices_series (np.ndarray): Simulated price paths of shape
            ``(n_steps, n_sims)``.
    """

    def __init__(
        self,
        s0: float,
        mu: float,
        sigma: float,
        n_steps: int = 100,
        n_sims: int = 1,
        stop_loss: float = 0,
        take_profit: float = np.inf,
    ):
        self.s0: float = s0
        self.mu: float = mu
        self.sigma: float = sigma
        self._epsilon: np.ndarray | None = None
        self.n_steps: int = n_steps
        self.n_sims: int = n_sims
        self.prices_series: np.ndarray | None = None
        self.stop_loss: float = stop_loss
        self.take_profit: float = take_profit

    def run_simulation(self) -> None:
        """Perform ``n_sims`` GBM simulations and store the price paths.

        The simulation uses the discretized GBM formula
        ``S_{t+1} = S_t * (1 + mu + sigma * epsilon)``, where
        ``epsilon`` is standard normal noise.
        """
        self._epsilon = np.random.randn(self.n_steps, self.n_sims)
        gross_prices = 1 + self.mu + (self.sigma * self._epsilon)
        self.prices_series = self.s0 * np.cumprod(gross_prices, axis=0)
    
    def apply_sl_tp(
        self, stop_loss: float = 0, take_profit: float = np.inf
    ) -> np.ndarray:
        """Apply stop‑loss/take‑profit rules to simulated paths.

        Parameters
        ----------
        stop_loss
            Price level triggering a stop loss exit.
        take_profit
            Price level triggering a take profit exit.

        Returns
        -------
        np.ndarray
            Price series modified so that once an exit level is reached the
            path remains constant thereafter (the exit price).
        """
        prices = self.prices_series.copy()

        mask_sl = prices < stop_loss
        mask_tp = prices >= take_profit
        mask_trigger = mask_sl | mask_tp

        first_exit = np.argmax(mask_trigger, axis=0)
        exit_prices = prices[first_exit, np.arange(self.n_sims)]
        closing_prices = np.where(exit_prices <= stop_loss, stop_loss, take_profit)

        persistent_mask = np.maximum.accumulate(mask_trigger, axis=0)

        return np.where(persistent_mask, closing_prices, prices)
    
    @property
    def expected_wealth(self) -> float:
        """Mean terminal price across all simulations."""
        return self.prices_series[-1].mean()
    
    @property
    def std_dev(self) -> float:
        """Standard deviation of terminal prices."""
        return self.prices_series[-1].std()
    
    @property
    def performance_ratio(self) -> float:
        """Simple performance metric defined as expected wealth over risk."""
        return self.expected_wealth / self.std_dev

    @property
    def loss_probability(self):
        loss = self.prices_series[-1] < self.s0
        return loss.mean()
    
    @property
    def var(self):
        return self.s0 - np.percentile(self.prices_series[-1], 5)
        
    def plot_simulation(self) -> None:
        """Visualize all simulated price paths."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.prices_series)
        plt.title(f"GBM simulation: {self.n_sims} paths")
        plt.xlabel("Time step")
        plt.ylabel("Price (S_t)")
        plt.grid(True)
        plt.show()

    def plot_result_distribution(self) -> None:
        """Show histogram of final prices across all simulations."""
        plt.figure(figsize=(10, 6))
        plt.hist(self.prices_series[-1], bins=max(10, self.n_sims // 10))
        plt.title(f"Distribution of terminal prices ({self.n_sims} sims)")
        plt.xlabel("Price (S_final)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()