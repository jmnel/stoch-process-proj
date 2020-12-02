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
print(DATA_PATH)

# random.seed(0)
# np.random.seed(0)

n = 5
n_ban = 2
pairs = {frozenset([i, j]) for i in range(n) for j in range(i + 1, n) if i != j}

#bans = set(random.sample(tuple(pairs), n_ban))

f = distributions.gamma(a=10)

#distances = {p: f.rvs() for p in pairs.difference(bans)}

# data = {'distances': distances,
#        'bans': bans}

# with open('data.pkl', 'wb') as fh:
#    pickle.dump(data, fh)

with open(DATA_PATH, 'rb') as fh:
    data = pickle.load(fh)

distances = data['distances']
bans = data['bans']


def legal(t):
    for i in range(n):
        if {t[i], t[(i + 1) % n]} in bans:
            return False
    return True


def dist(t):
    return sum(distances[frozenset((t[i], t[(i + 1) % n]))] for i in range(n))


# random.seed(time.perf_counter_ns())

n_samples = 100000

plt_ints = list(range(0, n_samples + 1, 1000))
plt_ints = plt_ints[1:]


d = 0.
n_legal = 0
plt_x = list()
plt_y = list()

for i_pts in plt_ints:

    while n_legal < i_pts:
        t = random.sample(range(n), n)

        if legal(t):

            d += dist(t)
            n_legal += 1

    plt_x.append(i_pts)
    plt_y.append(d / n_legal)

plt.plot(plt_x, plt_y, linewidth=0.4)
# plt.show()

est_avg_dist = d / n_legal
print(est_avg_dist)

all_tours = set()
while len(all_tours) < math.factorial(n):
    all_tours.add(tuple(random.sample(range(n), n)))

true_tours = {t for t in all_tours if legal(t)}

print(f'n true tours = {len(true_tours)}')
print(f'n all tours = {len(all_tours)}')

avg_dist_exact = sum([dist(t) for t in true_tours]) / len(true_tours)
print(f'exact={avg_dist_exact}')

t0 = tuple(random.sample(range(n), n))
while not legal(t0):
    t0 = tuple(random.sample(range(n), n))

print(f't0={t0}')

n_warmup = 1000
n_chain = 100000

t = t0
for k in range(n_warmup):
    i, j = random.sample(range(5), 2)
#    print(f'{k} -> {i},{j}, t={t}')
    t2 = list(t)
    t2[i], t2[j] = t2[j], t2[i]

    if legal(t2):
        t = tuple(t2)

d = 0.
plt_y = list()
for k in range(n_chain):
    i, j = random.sample(range(5), 2)
#    print(f'{k} -> {i},{j}, t={t}')
    t2 = list(t)
    t2[i], t2[j] = t2[j], t2[i]

    if legal(t2):
        t = tuple(t2)

    d += dist(t)
    plt_y.append(d / (k + 1))

print(f'dist_chain={d/n_chain}')

plt.plot(list(range(n_chain)), plt_y, linewidth=0.4)
plt.plot([0, max(n_chain, len(plt_x))], [avg_dist_exact, ] * 2, linewidth=1)
plt.ylim([57.5, 58])

plt.show()
