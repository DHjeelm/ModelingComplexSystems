import time
import numpy as np
import matplotlib.pyplot as plt
from random import random
from copy import deepcopy
from matplotlib.colors import ListedColormap
import matplotlib.patches as mPatches
from sympy import E

susceptible = 0
infected = 1
recovered = 2
immune = 3

timeToBecomeSusceptible = 15
timeToLooseImmunity = 150
maximumTimeToRecover = 20


probabilities = {"initiallyInfected": .01,
                 immune: 0.3, "recovery": .1, "infection": .3}


class Person:

    def __init__(self, state=susceptible) -> None:
        self.newState = None
        self.state = susceptible
        self.timeToSusceptible = 0
        self.timeToRecover = 0

    def infect(self):
        self.oldState = self.state
        self.state = infected
        self.timeToRecover = random() * maximumTimeToRecover

    def recover(self):
        self.state = recovered


class Population:
    def __init__(self, N: int) -> None:

        nrInfected = 0
        self.N = N
        self.individuals = [[Person() for _ in range(N)] for i in range(N)]
        for i in range(N):
            for j in range(N):
                if random() <= probabilities["initiallyInfected"]:
                    self.individuals[i][j].infect()
                    nrInfected += 1

        print(self.getStates())

        if nrInfected == 0:
            self.individuals[N//2][N//2].infect()
        print(self.getStates())

    def getStates(self):
        return [[person.state for person in persons] for persons in self.individuals]

    def applyNewState(self):
        for persons in self.individuals:
            for person in persons:
                if person.newState is not None:
                    person.state = person.newState


def updatePopulation(population: Population):
    N = population.N

    for i, row in enumerate(population.individuals):
        for j, person in enumerate(row):

            # check if person is immune
            if person.state == immune or person.state == recovered:
                if person.timeToSusceptible == 0:
                    person.newState = susceptible
                else:
                    person.timeToSusceptible -= 1

                continue

            # # Check if person has just recovered
            # if person.state == recovered:
            #     if person.timeToSusceptible == 0:
            #         person.newState = susceptible
            #     else:
            #         person.timeToSusceptible -= 1
            #     continue

            # Check if infected
            if person.state == infected and random() <= probabilities["recovery"]:
                if random() <= probabilities[immune]:
                    person.newState = immune
                    person.timeToSusceptible = timeToLooseImmunity

                else:
                    person.newState = recovered
                    person.timeToSusceptible = timeToBecomeSusceptible
                continue

            # Susceptible
            # check neighbors
            for r in range(-1, 2):
                for c in range(-1, 2):
                    if r == 0 and c == 0:
                        continue

                    rr = r + i if r + i < N else 0
                    cc = c + j if c + j < N else 0
                    if population.individuals[rr][cc].state == infected and random() <= probabilities["infection"]:
                        person.newState = infected
                        break
                else:
                    continue
                break
    population.applyNewState()


def plotPopulation(population: Population, title: str):

    cmap = ListedColormap(["white", "red", "yellow", "blue"])
    susceptible_patch = mPatches.Patch(color="white", label="Susceptible")
    infected_patch = mPatches.Patch(color="red", label="Infected")
    recovered_patch = mPatches.Patch(color="yellow", label="Recovered")
    immune_patch = mPatches.Patch(color="blue", label="Immune")
    neverInfected_patch = mPatches.Patch(color="green", label="Never Infected")

    # print(population.getStates())

    plt.figure(1)
    plt.title(title)
    plt.legend(handles=[infected_patch, susceptible_patch,
                        recovered_patch, immune_patch], bbox_to_anchor=(1.05, 1), loc=2)
    plt.imshow(population.getStates(),
               vmin=0, vmax=len(cmap.colors), cmap=cmap)
    plt.yticks(color="w")

    plt.show()


if __name__ == "__main__":

    N = 100

    population = Population(N)
    population.getStates()
    plotPopulation(population, "Initial state")

    nSteps = 400
    plt.ion()

    for i in range(nSteps):
        updatePopulation(population)
        if not np.any(np.array(population.getStates()) == 1):
            plt.close()
            plt.ioff()
            plotPopulation(population, f"Population cured after {i + 1} steps")
            break
        plt.close()
        plotPopulation(population, f"Population {i + 1}")
        plt.pause(0.01)

    # plotPopulation(population, "Final state")
