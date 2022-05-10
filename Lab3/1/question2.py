import multiprocessing
from threading import Thread
from question1 import *
import numpy as np
import random
from multiprocessing import Process, Lock
import time
import sys
from functools import partial
from itertools import repeat
import ctypes


q = .01
r = .01
N = 1000
T = 2000


nSimulations = 20


def worker(data: list, resultToWrite):

    for info in data:
        i = info[0]
        p = info[1]
        for j in range(nSimulations):

            population = initialiazePopulation(N)

            for t in range(T):
                population = updatePopulation(population, p, q, r)

            nSharers = np.count_nonzero(population == sharer)
            resultToWrite[i * nSimulations + j] = nSharers

        progress = np.sum(np.array(resultToWrite) >= 0) / len(resultToWrite)
        print(f"{progress * 100}% done")

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

    ps = np.linspace(0, .001, 40)

    progress = 0

    nProcesses = multiprocessing.cpu_count() // 2
    print(f"Starting process with {nProcesses} workers")
    start = time.time()
    preparedResult = -1 * np.ones(
        (nSimulations * ps.size), dtype=ctypes.c_int)
    shared_array = to_shared_array(preparedResult, ctype=ctypes.c_int)
    result = to_numpy_array(shared_array, preparedResult.shape)

    work = split(list(enumerate(ps)), nProcesses)
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

    ps = ps.repeat(nSimulations)
    print(ps.shape)
    print(result.shape)

    stop = time.time()
    print(f"Execution time: {stop - start} seconds")

    plt.figure(1)
    plt.scatter(ps, result)
    plt.xlabel("value of q")
    plt.ylabel(f"Number of sharers after {T} time steps")
    plt.savefig("Scatter_p.png")

    plt.figure(2)
    plt.hist2d(ps, result, bins=10)
    plt.xlabel("value of p")
    plt.ylabel(f"Number of sharers after {T} time steps")
    plt.colorbar()
    plt.savefig("Heatmap_p.png")

    plt.show()
