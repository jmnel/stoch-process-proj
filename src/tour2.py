import random
from pprint import pprint
import time
import json
import pickle
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
import scipy.stats.distributions as distributions

DATA_PATH = Path(__file__).absolute().parents[1] / 'data' / 'data.pkl'
# random.seed(0)
# np.random.seed(0)

n_cities = 20
n_ban = 5
pairs = {frozenset([i, j]) for i in range(n_cities) for j in range(i + 1, n_cities) if i != j}

gamma = distributions.gamma(a=10)
bans = random.sample(pairs, n_ban)

distances = {p: gamma.rvs() for p in pairs.difference(bans)}


# data = {'distances': distances,
#        'bans': bans}

# with open('data.pkl', 'wb') as fh:
#    pickle.dump(data, fh)

with open(DATA_PATH, 'rb') as fh:
    data = pickle.load(fh)

distances = data['distances']
bans = data['bans']


def legal(t):
    for i in range(n_cities):
        if {t[i], t[(i + 1) % n_cities]} in bans:
            return False
    return True


def dist(t):
    return sum(distances[frozenset((t[i], t[(i + 1) % n_cities]))] for i in range(n_cities))


#m = 100000
#n = 2000


def mcmc(n_burn, m_samples):

    # Pick random initial state; repeat until we have legal first tour.
    t0 = tuple(random.sample(range(n_cities), n_cities))
    while not legal(t0):
        t0 = tuple(random.sample(range(n_cities), n_cities))

    # Do burn-in.
    t = t0
    for k in range(n):
        i, j = random.sample(range(n_cities), 2)

        t_prime = list(t)
        t_prime[i], t_prime[j] = t_prime[j], t_prime[i]
        t_prime = tuple(t_prime)
        if legal(t_prime):
            t = t_prime

    # Sample MCMC.
    d = 0.
    d += dist(t)
    running_avg = [d, ]
    for k in range(m):
        i, j = random.sample(range(n_cities), 2)

        t_prime = list(t)
        t_prime[i], t_prime[j] = t_prime[j], t_prime[i]
        t_prime = tuple(t_prime)
        if legal(t_prime):
            t = t_prime

        d += dist(t)
        running_avg.append(d / (k + 2))

    return d / (m + 1), running_avg


n = 100000
m = 100

num_runs = 10

for i in range(num_runs):
    d_exp, plt_d_exp = mcmc(n, m)
    plt.plot(list(range(len(plt_d_exp))), plt_d_exp)


plt.show()

print(f'E[f(t)] = {d_exp}')
