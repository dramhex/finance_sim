from simulator import GBMSimulator

sim_bitcoin = GBMSimulator(s0=50000, mu=0.002, sigma=0.05, n_steps=100, n_sims=1000, stop_loss=49000, take_profit=55000)

sim_bitcoin.run_simulation()

print("expected wealth", sim_bitcoin.expected_wealth)
print("std deviation", sim_bitcoin.std_dev)
print("ratio", sim_bitcoin.ratio)

print("probability of loss", sim_bitcoin.loss_probability)

print("value at risk :", sim_bitcoin.var)

sim_bitcoin.plot_simulation()

sim_bitcoin.plot_result_distribution()