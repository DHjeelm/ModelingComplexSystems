import multiprocessing
from threading import Thread
from requests import put
from sympy import nsimplify
from question1 import *
import numpy as np
import random
import seaborn as sns
from multiprocessing import Process, Lock
import time
import sys
from functools import partial
from itertools import repeat
import ctypes


N = 1000
T = 4000
nSimulations = 20

p = .001
r = .01

qs = np.linspace(0, .1, 40)

progress = 0


def worker(data: list, resultToWrite: np.ndarray):

    for info in data:
        i = info[0]
        q = info[1]
        for j in range(nSimulations):

            population = initialiazePopulation(N)

            for t in range(T):
                population = updatePopulation(population, p, q, r)

            nSharers = np.count_nonzero(population == sharer)
            resultToWrite[i * nSimulations + j] = nSharers

        progress = (i * nSimulations + j) / len(resultToWrite)
        print(f"{progress}% done")

        sys.stdout.flush()


def to_shared_array(arr, ctype):
    shared_array = multiprocessing.Array(ctype, arr.size, lock=False)
    temp = np.frombuffer(shared_array, dtype=arr.dtype)
    temp[:] = arr.flatten(order='C')
    return shared_array


def to_numpy_array(shared_array, shape):
    '''Create a numpy array backed by a shared memory Array.'''
    arr = np.ctypeslib.as_array(shared_array)
    return arr.reshape(shape)


def split(a, n):
    k, m = divmod(len(a), n)
    return list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))


if __name__ == "__main__":
    nProcesses = 6
    start = time.time()
    preparedResult = np.ndarray(
        (nSimulations * qs.size), dtype=ctypes.c_int)
    shared_array = to_shared_array(preparedResult, ctype=ctypes.c_int)
    result = to_numpy_array(shared_array, preparedResult.shape)

    work = split(list(enumerate(qs)), nProcesses)
    processes = []

    for process in range(nProcesses):
        data = work[process]
        process = Process(target=worker, args=(data, shared_array))
        processes.append(
            process)
        process.start()

    for process in processes:
        process.join()

        # for i, q in enumerate(qs):
        #     for j in range(nSimulations):

        #         population = initialiazePopulation(N)

        #         for t in range(T):
        #             population = updatePopulation(population, p, q, r)

        #         nSharers = np.count_nonzero(population == sharer)
        #         result[i * nSimulations + j] = nSharers
        #     progress = (i * nSimulations + j) / result.size
        #     print(f"{progress}% done")

    qs = qs.repeat(nSimulations)
    print(qs.shape)
    print(result.shape)

    stop = time.time()
    print(f"Execution time: {stop - start} seconds")

    plt.figure(1)
    plt.scatter(qs, result)
    plt.xlabel("value of q")
    plt.ylabel(f"Number of sharers after {T} time steps")
    plt.savefig("Scatter.png")

    plt.figure(2)
    plt.hist2d(qs, result, bins=10)
    plt.xlabel("value of q")
    plt.ylabel(f"Number of sharers after {T} time steps")
    plt.colorbar()
    plt.savefig("Heatmap.png")

    plt.show()
