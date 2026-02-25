import numpy as np
import matplotlib.pyplot as plt

class LSTPOptimizer:
    def __init__(self, simulator, sl_min=0.1, sl_max=0.95, tp_min=1.05, tp_max=5, step=0.05):
        self.sl_range = np.arange(sl_min, sl_max+step, step)
        self.tp_range = np.arange(tp_min, tp_max+step, step)
        self.simulator = simulator
        self.SL_grid, self.TP_grid = np.meshgrid(self.sl_range, self.tp_range)

        self.returns = None
        self.mean_ret = None
        self.std_ret = None
        self.sharpe_ratio = None

    def optimize(self):
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

        self.returns = (final_wealth - self.simulator.s0) / self.simulator.s0
        self.mean_ret = self.returns.mean(axis=0)  # Moyenne par stratégie
        self.std_ret = self.returns.std(axis=0)    # Risque par stratégie

        # mean_ret_2d = mean_ret.reshape(SL_grid.shape)
        # plt.imshow(mean_ret_2d, extent=[0.5, 0.95, 1.05, 2.0], origin="lower")
        # plt.colorbar(label="Rendement Moyen")
        # plt.xlabel("Stop Loss (fraction du prix)")
        # plt.ylabel("Take Profit (fraction du prix)")
        # plt.title("Optimisation de la stratégie SL/TP sur Bitcoin")
        # plt.show()
    
    def plot_sharpe_ratio(self):
        sharpe_ratio = self.mean_ret / self.std_ret
        sharpe_ratio_2d = sharpe_ratio.reshape(self.SL_grid.shape)
        plt.imshow(sharpe_ratio_2d, extent=[self.sl_range.min(), self.sl_range.max(), self.tp_range.min(), self.tp_range.max()], origin="lower")
        plt.colorbar(label="Ratio de Sharpe")
        plt.xlabel("Stop Loss (fraction du prix)")
        plt.ylabel("Take Profit (fraction du prix)")
        plt.title("Optimisation de la stratégie SL/TP sur Bitcoin")
        plt.show()