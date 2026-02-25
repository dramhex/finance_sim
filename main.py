from simulator import GBMSimulator
from optimizer import LSTPOptimizer
import matplotlib.pyplot as plt
import numpy as np

def main() -> None:
    """Example script demonstrating simulator and optimizer usage."""

    # simulate bitcoin trajectories
    sim_bitcoin = GBMSimulator(s0=50000, mu=0.002, sigma=0.05, n_steps=100, n_sims=1000)
    sim_bitcoin.run_simulation()

    # visualizations of raw simulation
    sim_bitcoin.plot_simulation()
    sim_bitcoin.plot_result_distribution()

    # optimize SL/TP grid and display metrics
    optimizer = LSTPOptimizer(
        sim_bitcoin, sl_min=0.5, sl_max=0.95, tp_min=1.05, tp_max=4, step=0.05, fee_rate=0.005
    )
    best_sl, best_tp = optimizer.optimize()
    print(
        f"Optimal SL={best_sl:.3f}, TP={best_tp:.3f}, win rate={optimizer.best_win_rate:.2%}"
    )

    # plot results from optimizer
    optimizer.plot_sharpe_ratio()
    optimizer.plot_optimal_equity_curve()


if __name__ == "__main__":
    main()