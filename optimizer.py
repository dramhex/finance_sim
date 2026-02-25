import numpy as np
import matplotlib.pyplot as plt

class LSTPOptimizer:
    # simple optimizer that looks for the best stop-loss / take-profit
    def __init__(self, simulator, sl_min=0.1, sl_max=0.95,
                 tp_min=1.05, tp_max=5, step=0.05, fee_rate=0.005):
        self.sl_range = np.arange(sl_min, sl_max+step, step)
        self.tp_range = np.arange(tp_min, tp_max+step, step)
        self.simulator = simulator
        self.SL_grid, self.TP_grid = np.meshgrid(self.sl_range, self.tp_range)

        # will be computed later
        self.returns = None
        self.mean_ret = None
        self.std_ret = None
        self.sharpe_ratio = None

        # best strategy parameters
        self.best_returns = None
        self.idx_best = None
        self.best_sl = None
        self.best_tp = None
        self.fee_rate = fee_rate
        self.average_equity = None

    def calculate_metrics(self):
        # compute returns and sharpe for each grid point
        sl_flat = self.SL_grid.flatten() * self.simulator.s0
        tp_flat = self.TP_grid.flatten() * self.simulator.s0

        prices = self.simulator.prices_series[:, :, np.newaxis]

        mask_sl = prices <= sl_flat
        mask_tp = prices >= tp_flat
        triggered = mask_sl | mask_tp

        exit_idx = np.argmax(triggered, axis=0)
        any_trig = triggered.any(axis=0)
        exit_idx = np.where(any_trig, exit_idx, -1)

        sims = np.arange(self.simulator.n_sims)[:, None]
        final = self.simulator.prices_series[exit_idx, sims]

        self.returns = ((final - self.simulator.s0) / self.simulator.s0) - self.fee_rate
        self.mean_ret = self.returns.mean(axis=0)
        self.std_ret = self.returns.std(axis=0)
        self.sharpe_ratio = self.mean_ret / self.std_ret

    def get_optimized_params(self):
        if self.sharpe_ratio is None:
            self.calculate_metrics()
        self.idx_best = int(np.argmax(self.sharpe_ratio))
        self.best_sl = self.SL_grid.flatten()[self.idx_best]
        self.best_tp = self.TP_grid.flatten()[self.idx_best]
        self.best_returns = self.returns[:, self.idx_best]
        return self.best_sl, self.best_tp

    def optimize(self):
        # alias
        return self.get_optimized_params()

    @property
    def best_win_rate(self):
        return (self.best_returns > 0).mean()

    def calculate_drowdown(self) -> np.ndarray:
        peak = np.maximum.accumulate(self.average_equity)
        drawdown = (peak - self.average_equity) / peak
        # Le pire scénario historique en pourcentage
        max_dd = np.max(drawdown)
        print(f"Maximum Drawdown : {max_dd * 100:.2f}%")
        return drawdown

    def plot_sharpe_ratio(self):
        data = self.sharpe_ratio.reshape(self.SL_grid.shape)
        plt.imshow(data,
                   extent=[self.sl_range.min(), self.sl_range.max(),
                           self.tp_range.min(), self.tp_range.max()],
                   origin="lower")
        plt.colorbar(label="Sharpe ratio")
        plt.xlabel("SL fraction")
        plt.ylabel("TP fraction")
        plt.title("Sharpe heatmap")
        plt.show()

    def plot_optimal_equity_curve(self):
        eq = self.simulator.apply_sl_tp(self.best_sl * self.simulator.s0,
                                        self.best_tp * self.simulator.s0)
        self.average_equity = eq.mean(axis=1)
        plt.plot(self.average_equity)
        plt.axhline(self.simulator.s0, color="red", linestyle="--")
        plt.title("Average equity (optimal)")
        plt.show()