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

from graphviz import Graph

DATA_PATH = Path(__file__).absolute().parents[1] / 'data' / 'data.pkl'
# random.seed(0)
# np.random.seed(0)

n_cities = 20
n_ban = 5
pairs = {frozenset([i, j]) for i in range(n_cities) for j in range(i + 1, n_cities) if i != j}

#bans = random.sample(pairs, n_ban)


# data = {'distances': distances,
#        'bans': bans}

# with open('data.pkl', 'wb') as fh:
#    pickle.dump(data, fh)

with open(DATA_PATH, 'rb') as fh:
    data = pickle.load(fh)

distances = data['distances']
bans = data['bans']

paths = {s for s in pairs if s not in bans}

dot = Graph()

for i in range(n_cities):
    dot.node(f'{i}')

for p in paths:
    i, j = tuple(p)
    dot.edge(f'{i}', f'{j}')
#    dot.edge(f'{j}', f'{i}')

dot.render(view=True)
