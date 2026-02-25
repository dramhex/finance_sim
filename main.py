from simulator import GBMSimulator

sim_bitcoin = GBMSimulator(s0=50000, mu=0.002, sigma=0.05, n_steps=100, n_sims=300)

# Afficher les simulations
sim_bitcoin.plot_simulation()