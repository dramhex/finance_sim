from simulator import GBMSimulator
import numpy as np
import matplotlib.pyplot as plt

sim_bitcoin = GBMSimulator(s0=50000, mu=0.002, sigma=0.05, n_steps=100, n_sims=1000)

sim_bitcoin.run_simulation()

sl_range = np.arange(0.5, 1, 0.05)
tp_range = np.arange(1.05, 2.05, 0.05)

SL_grid, TP_grid = np.meshgrid(sl_range, tp_range)

sl_flat = SL_grid.flatten()
tp_flat = TP_grid.flatten()

prices_3d = sim_bitcoin.prices_series[:, :, np.newaxis]

mask_sl = prices_3d <= (sl_flat * sim_bitcoin.s0)
mask_tp = prices_3d >= (tp_flat * sim_bitcoin.s0)
mask_trigger = mask_sl | mask_tp

exit_indices = np.argmax(mask_trigger, axis = 0)

# On identifie si une sortie a eu lieu
any_trigger = mask_trigger.any(axis=0) # (n_sims, n_strategies)

# Si ça a déclenché, on garde l'indice. Sinon, on prend le dernier (-1)
actual_exit_indices = np.where(any_trigger, exit_indices, -1)

# On crée un vecteur pour aligner les simulations (0, 1, 2...)
sim_indices = np.arange(sim_bitcoin.n_sims)[:, np.newaxis]

# LA MAGIE : On extrait les prix au moment des sorties pour tout le monde d'un coup
# On obtient une matrice (n_sims, n_strategies)
final_wealth = sim_bitcoin.prices_series[actual_exit_indices, sim_indices]

returns = (final_wealth - sim_bitcoin.s0) / sim_bitcoin.s0
mean_ret = returns.mean(axis=0)  # Moyenne par stratégie
std_ret = returns.std(axis=0)    # Risque par stratégie

print(returns)
print(mean_ret)
print(std_ret)

mean_ret_2d = mean_ret.reshape(SL_grid.shape)

# plt.imshow(mean_ret_2d, extent=[0.5, 0.95, 1.05, 2.0], origin="lower")
# plt.colorbar(label="Rendement Moyen")
# plt.xlabel("Stop Loss (fraction du prix)")
# plt.ylabel("Take Profit (fraction du prix)")
# plt.title("Optimisation de la stratégie SL/TP sur Bitcoin")
# plt.show()

sharpe_ratio = mean_ret / std_ret
sharpe_ratio_2d = sharpe_ratio.reshape(SL_grid.shape)
plt.imshow(sharpe_ratio_2d, extent=[0.5, 0.95, 1.05, 2.0], origin="lower")
plt.colorbar(label="Ratio de Sharpe")
plt.xlabel("Stop Loss (fraction du prix)")
plt.ylabel("Take Profit (fraction du prix)")
plt.title("Optimisation de la stratégie SL/TP sur Bitcoin")
plt.show()