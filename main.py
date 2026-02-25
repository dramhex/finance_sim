from simulator import GBMSimulator
from optimizer import LSTPOptimizer

sim_bitcoin = GBMSimulator(s0=50000, mu=0.002, sigma=0.05, n_steps=100, n_sims=1000)
sim_bitcoin.run_simulation()

sim_bitcoin.plot_simulation()
sim_bitcoin.plot_result_distribution()

prices_with_sl_tp = sim_bitcoin.apply_sl_tp(stop_loss=40000, take_profit=70000)

optimizer = LSTPOptimizer(sim_bitcoin, sl_min=0.5, sl_max=0.95, tp_min=1.05, tp_max=2.0)
optimizer.optimize()
optimizer.plot_sharpe_ratio()