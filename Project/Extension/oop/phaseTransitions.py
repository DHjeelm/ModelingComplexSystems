import time
from concurrent.futures import thread
import ctypes
from fcntl import flock
from unittest import case
import matplotlib.pyplot as plt
from simulation import Simulation
import multiprocessing
from typing import List
import numpy as np
import sys


nSimulations = 50


def worker(threadId: int, flockingSettings: List[bool], resultToWrite):

    for j in range(nSimulations):

        simulation = Simulation(nPrey=50, nPred=3)
        simulation.simulate(0.01, 5)
        simulation.analysisHelper(flockingSettings)

        nDeadPrey = simulation.countEatenPrey()
        resultToWrite[threadId * nSimulations + j] = nDeadPrey

    progress = np.sum(np.array(resultToWrite) > 0) / len(resultToWrite)
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


def getCombinations(items):
    from itertools import combinations
    list_combinations = list()
    for n in range(len(items) + 1):
        list_combinations += list(combinations(items, n))
    # print(list_combinations)
    return list_combinations


def getTitles(values):

    text = ""

    if 1 in values:
        text += "Align\n"

    if 2 in values:
        text += "Cohesion\n"

    if 3 in values:
        text += "Separation"

    if text == "":
        text = "No flocking"

    return text


def flockBehavior():

    timeStep = 0.01
    T = 4

    cases = getCombinations([1, 2, 3])
    nWorkers = len(cases)

    preparedResult = np.zeros(
        (nSimulations * nWorkers), dtype=ctypes.c_int)
    shared_array = to_shared_array(preparedResult, ctype=ctypes.c_int)
    result = to_numpy_array(shared_array, preparedResult.shape)

    workers = []

    start = time.time()
    for i in range(nWorkers):
        flockingSettings = cases[i]

        process = multiprocessing.Process(
            target=worker, args=(i, flockingSettings, shared_array))
        process.start()
        workers.append(process)

    for process in workers:
        process.join()

    labels = np.array([getTitles(_case)
                       for _case in cases]).repeat(nSimulations)
    print(labels.shape)
    # print(result.shape)

    # print(labels)
    # print(result)

    stop = time.time()
    print(f"Execution time: {stop - start} seconds")

    plt.figure(1)
    plt.scatter(labels, result)
    plt.xlabel("Flocking behaviour")
    plt.ylabel(f"Number of eaten prey")
    plt.savefig("Scatter_p.png")

    plt.figure(2)
    plt.xticks(np.array(list(range(nWorkers))).repeat(
        nSimulations), labels)
    plt.hist2d(np.array(list(range(nWorkers))).repeat(
        nSimulations), result, bins=len(cases))
    plt.xlabel("Flocking behaviour")
    plt.ylabel(f"Number of eaten prey")
    plt.colorbar()
    plt.savefig("Heatmap_p.png")

    plt.show()

    print(result)


if __name__ == "__main__":
    flockBehavior()
