import random
from pprint import pprint
import time
import json
import pickle
import math
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')
mpl.use('Agg')
import scipy.stats.distributions as distributions

DATA_PATH = Path(__file__).absolute().parents[1] / 'data' / 'data.pkl'

# Make results reproducible.
random.seed(0)
np.random.seed(0)

n_cities = 20
n_ban = 5

# Enumerate edges between pairs of cities.
pairs = {frozenset([i, j]) for i in range(n_cities) for j in range(i + 1, n_cities) if i != j}

bans = random.sample(pairs, n_ban)


# Define and plot gamma distribution.
gamma = distributions.gamma(a=7.5)
fig, ax = plt.subplots(1)
x = np.linspace(0, 25, 100)
z = gamma.pdf(x)
ax.plot(x, z)
ax.set_xlabel('Distance')
ax.set_ylabel('Probability')
fig.savefig('fig1.pdf', dpi=200)

# Randomly sample distances for each edge.
distances = {p: gamma.rvs() for p in pairs.difference(bans)}

# Create LaTeX table for distances.
with open('table1-1.tex', 'wt') as fh:

    vals = list((i, j, d) for (i, j), d in distances.items())
    vals = sorted(vals, key=lambda e: (e[0], e[1]))

    row = ' & '.join(('{} & {} & {:.2f}',) * 5) + ' \\\\\n'

    for i in range(len(vals) // 5):
        row_s = row.format(*vals[i], *vals[i + 37], *vals[i + 2 * 37], *vals[i + 3 * 37], *vals[i + 4 * 37])
        fh.write(row_s)

# Create LaTeX table for forbidden eges.
with open('table1-2.tex', 'wt') as fh:

    vals = list((i, j) for (i, j) in bans)
    vals = sorted(vals, key=lambda e: (e[0], e[1]))

    for e in vals:
        fh.write('{} & {} \\\\\n'.format(*e))


def legal(t):
    for i in range(n_cities):
        if {t[i], t[(i + 1) % n_cities]} in bans:
            return False
    return True


def dist(t):
    return sum(distances[frozenset((t[i], t[(i + 1) % n_cities]))] for i in range(n_cities))


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

        if k % 100 == 0:
            running_avg.append(d / (k + 2))

    return d / (m + 1), running_avg


n = 100000
m = 1000000

num_runs = 10

fig, ax = plt.subplots(1, 1)

for i in range(num_runs):
    d_exp, plt_d_exp = mcmc(n, m)
    ax.plot(list(range(len(plt_d_exp))), plt_d_exp, linewidth=1.0)
    print(f'{i} -> E[f(t)] = {d_exp}')

ax.set_ylabel('$E[ f(x) ]$ estimate')
ax.set_xlabel('$m$')

fig.savefig('fig2.pdf', dpi=200)
