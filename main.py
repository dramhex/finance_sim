from simulator import GBMSimulator
from optimizer import LSTPOptimizer

sim_bitcoin = GBMSimulator(s0=50000, mu=0.002, sigma=0.05, n_steps=100, n_sims=1000)
sim_bitcoin.run_simulation()

sim_bitcoin.plot_simulation()
sim_bitcoin.plot_result_distribution()

prices_with_sl_tp = sim_bitcoin.apply_sl_tp(stop_loss=40000, take_profit=70000)

optimizer = LSTPOptimizer(sim_bitcoin, sl_min=0.5, sl_max=0.95, tp_min=1.05, tp_max=4, step=0.05, fee_rate=0.005)
optimizer.calculate_metrics()
optimizer.get_optimized_params()
print(optimizer.best_win_rate)
print(optimizer.best_sl, optimizer.best_tp)