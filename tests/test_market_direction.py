import sys
import os 
import matplotlib.pyplot as plt 
import numpy as np 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from simulation.market_gym import Market
from multiprocessing import Pool
from numpy.random import SeedSequence
import time 

# test the directional trend of the market, 
# Do 1e4 iterations of 1000 steps. Calculate the average drift in the market, measured by mid_price_end - mid_price_start. 
# do to: multiprocessing evaluation. increase speed. 

# Results: 
# drift seems to be ok: 

# sample_from_environment(0)
# ToDo: There seems to be an upward trend in the environment. Investigate why this happens. 
# First guess, is that it is due to the initial conditions. Orderbook is slightly out of balance.
# However: the drift is not very strong over 1000 steps.

# actually, for 10000 times 1000 steps, the drift is sliqhtly negative. so seems to be ok,
# in general, run even more sanity checks. 

# for 70.000 runs of the 1000 step environment the drift is around -0.007 or similar. so seems to be ok.


def sample_from_environment(seed, n_env_steps=1000, n_steps=1000):
    config = {
        'market_env': 'noise',
        'execution_agent': 'rl_agent',
        'volume': 10,
        'seed': seed,
        'terminal_time': 150,
        'time_delta': 15,
        'drop_feature': None,
    }
    differences = []
    M = Market(config)
    for _ in range(n_steps):
        M.reset()
        initial_mid_price = (M.get_best_price('bid') + M.get_best_price('ask')) / 2
        for _ in range(n_env_steps):
            M.generate_order()
        final_mid_price = (M.get_best_price('bid') + M.get_best_price('ask')) / 2
        differences.append(final_mid_price - initial_mid_price)
    return differences


def run_market_direction_study(n_workers=70):
    # set n workers, and seeds
    seeds = [s + 10 for s in range(n_workers)]
    t = time.time()
    p = Pool(n_workers)
    out = p.map(sample_from_environment, seeds)
    print(f'time for simulation: {time.time()-t}s')
    differences = np.concatenate(out)
    # results
    print(f'total number of environment steps {len(differences)}')
    print(f'average drift {np.mean(differences)}')
    plt.hist(differences, bins=np.arange(start=-4, stop=5, step=0.5))
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/market_direction.png')


if __name__ == '__main__':
    run_market_direction_study()
            





