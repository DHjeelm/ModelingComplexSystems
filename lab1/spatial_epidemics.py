

import numpy as np
import matplotlib.pyplot as plt
import copy
from random import random
from matplotlib.colors import ListedColormap
import matplotlib.patches as mPatches
import time


# Define grid size
N = 100


def initializePopulation(N: int):
    # 0 is susceptible
    # 1 is infected
    population = np.zeros((1, N), dtype=int)
    population[0, N//2] = 1
    return population


population = initializePopulation(N)

print(population)


def updateState(population, gamma):
    newState = copy.deepcopy(population)

    for i in range(len(population[0])):
        # Rule for infected persons
        if population[0, i] == 1 and random() <= 1-gamma:
            newState[0, i] = 0
            continue
        # Rule for susceptible
        # print(f"i: {i}\nvalue: {population[0, i]}")
        if i < N-1:

            if population[0, i-1] == 1 or population[0, i+1] == 1:
                if random() <= 1-gamma:
                    newState[0, i] = 1
                    # print("Infected\n")
        else:
            if population[0, i-1] == 1 or population[0, 0] == 1:
                if random() <= 1-gamma:
                    newState[0, i] = 1
                    # print("Infected\n")
    return newState


cmap = ListedColormap(["white", "red"])
susceptible_patch = mPatches.Patch(color="white", label="Susceptible")
infected_patch = mPatches.Patch(color="red", label="Infected")

# function to plot population


def plotPopulation(population, title):
    plt.figure(1)
    plt.title(title)
    plt.legend(handles=[infected_patch, susceptible_patch], loc="lower left")
    plt.imshow(population, vmin=0, vmax=len(cmap.colors), cmap=cmap)
    plt.yticks(color="w")
    plt.show()


plt.ion()
for i in range():
    population = updateState(population, .6)
    plotPopulation(population, "test")
    plt.pause(.5)


exit(0)


# Define gamma
gamma = {.6: 0, .5: 0, .4: 0, .3: 0}

chanceOfSurvival = []
nSims = 1000
nSteps = 1


for g, sum in gamma.items():

    population = initializePopulation(N)
    for i in range(nSims):
        for j in range(nSteps):
            population = updateState(population, g)

            if not np.all(population[0, :] == 0):
                sum += 1

    print(f"Sum: {sum}")
    gamma[g] = sum


print(gamma)
plt.plot(gamma.keys(), gamma.values())
plt.show()
