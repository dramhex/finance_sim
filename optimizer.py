import numpy as np
import matplotlib.pyplot as plt

class LSTPOptimizer:
    def __init__(self, simulator, sl_min=0.1, sl_max=0.95, tp_min=1.05, tp_max=5, step=0.05, fee_rate=0.005):
        self.sl_range = np.arange(sl_min, sl_max+step, step)
        self.tp_range = np.arange(tp_min, tp_max+step, step)
        self.simulator = simulator
        self.SL_grid, self.TP_grid = np.meshgrid(self.sl_range, self.tp_range)

        self.returns = None
        self.mean_ret = None
        self.std_ret = None
        self.sharpe_ratio = None

        self.best_returns = None
        self.idx_best = None
        self.best_sl = None
        self.best_tp = None
        self.fee_rate = fee_rate

    def calculate_metrics(self):
        sl_flat = self.SL_grid.flatten() * self.simulator.s0
        tp_flat = self.TP_grid.flatten() * self.simulator.s0

        prices_3d = self.simulator.prices_series[:, :, np.newaxis]

        mask_sl = prices_3d <= sl_flat
        mask_tp = prices_3d >= tp_flat
        mask_trigger = mask_sl | mask_tp

        exit_indices = np.argmax(mask_trigger, axis = 0)
        # On identifie si une sortie a eu lieu
        any_trigger = mask_trigger.any(axis=0) # (n_sims, n_strategies)
        # Si ça a déclenché, on garde l'indice. Sinon, on prend le dernier (-1)
        actual_exit_indices = np.where(any_trigger, exit_indices, -1)

        # On crée un vecteur pour aligner les simulations (0, 1, 2...)
        sim_indices = np.arange(self.simulator.n_sims)[:, np.newaxis]
        # On obtient une matrice (n_sims, n_strategies)
        final_wealth = self.simulator.prices_series[actual_exit_indices, sim_indices]

        self.returns = ((final_wealth - self.simulator.s0) / self.simulator.s0) - self.fee_rate
        self.mean_ret = self.returns.mean(axis=0)  # Moyenne par stratégie
        self.std_ret = self.returns.std(axis=0)    # Risque par stratégie
        self.sharpe_ratio = self.mean_ret / self.std_ret  # Ratio de Sharpe

    def get_optimized_params(self):
        self.calculate_metrics()

        self.idx_best = np.argmax(self.sharpe_ratio)
        self.best_sl = self.SL_grid.flatten()[self.idx_best]
        self.best_tp = self.TP_grid.flatten()[self.idx_best]
        self.best_returns = self.returns[:, self.idx_best]

        return self.best_sl, self.best_tp

        # mean_ret_2d = mean_ret.reshape(SL_grid.shape)
        # plt.imshow(mean_ret_2d, extent=[0.5, 0.95, 1.05, 2.0], origin="lower")
        # plt.colorbar(label="Rendement Moyen")
        # plt.xlabel("Stop Loss (fraction du prix)")
        # plt.ylabel("Take Profit (fraction du prix)")
        # plt.title("Optimisation de la stratégie SL/TP sur Bitcoin")
        # plt.show()
    
    @property
    def best_win_rate(self):
        return (self.best_returns > 0).mean()

    def plot_sharpe_ratio(self):
        sharpe_ratio_2d = self.sharpe_ratio.reshape(self.SL_grid.shape)
        plt.imshow(sharpe_ratio_2d, extent=[self.sl_range.min(), self.sl_range.max(), self.tp_range.min(), self.tp_range.max()], origin="lower")
        plt.colorbar(label="Ratio de Sharpe")
        plt.xlabel("Stop Loss (fraction du prix)")
        plt.ylabel("Take Profit (fraction du prix)")
        plt.title("Optimisation de la stratégie SL/TP sur Bitcoin")
        plt.show()