import random
from pprint import pprint
import time
import json
import math

n_cities = 20
n_ban = 5
pairs = {frozenset([i, j]) for i in range(n_cities) for j in range(i + 1, n_cities) if i != j}

bans = set(random.sample(pairs, n_ban))


swaps = [(i, j) for i in range(n_cities) for j in range(i + 1, n_cities)]

print(swaps)


def legal(t):

    for i in range(n_cities):
        if {t[i], t[(i + 1) % n_cities]} in bans:
            return False

    return True


t0 = random.sample(range(n_cities), n_cities)
while not legal(t0):
    t0 = random.sample(range(n_cities), n_cities)


t = t0
for k in range(100):
    t_new = list()
    g = 0
    b = 0
    cand = list()
    cand.append(t)
    for i, j in swaps:
        t_p = list(t)
        t_p[i], t_p[j] = t_p[j], t_p[i]

        if legal(t_p):
            g += 1
            cand.append(t_p)
        else:
            b += 1

    t = random.choice(cand)

    print(f'{k} -> legal={g} : bad={b}')
