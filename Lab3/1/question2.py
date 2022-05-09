from sympy import nsimplify
from question1 import *
import numpy as np
import random
import seaborn as sns
import time


if __name__ == "__main__":
    start = time.time()
    N = 100
    T = 4000
    nSimulations = 50

    p = .001
    r = .01

    qs = np.linspace(0, .1, 50)

    result = np.linspace(0, 1, nSimulations * qs.size)

    for i, q in enumerate(qs):
        for j in range(nSimulations):

            population = initialiazePopulation(N)

            for t in range(T):
                population = updatePopulation(population, p, q, r)

            nSharers = np.count_nonzero(population == sharer)
            result[i * nSimulations + j] = nSharers


qs = qs.repeat(nSimulations)
print(qs.shape)
print(result.shape)

stop = time.time()
print(f"Execution time: {stop - start} seconds")


plt.figure(1)
plt.scatter(qs, result)
plt.xlabel("value of q")
plt.ylabel(f"Number of sharers after {T} time steps")
plt.show()

plt.figure(2)
plt.hist2d(qs, result, bins=5)
plt.xlabel("value of q")
plt.ylabel(f"Number of sharers after {T} time steps")
plt.colorbar()


plt.show()
