# Imports
import numpy as np
import matplotlib.pyplot as plt
import copy
from random import random
from matplotlib.colors import ListedColormap
import matplotlib.patches as mPatches
import random

from sklearn import neighbors


# State to number representation
resting = 0
sharer = 1
bored = 2

# 0: Resting
# 1: Sharing
# 2: Bored

# Initiliaze population
def initialiazePopulation(N: int):

    # Create population
    population = np.zeros((N, N), dtype=int)


    # Pick two random indices
    randomPeople = random.sample(range(1, N), 9)

    # Set one to sharing
    population[randomPeople[0], randomPeople[1]] = 1

    # And the other one to bored
    population[randomPeople[2], randomPeople[3]] = 2
    # population[randomPeople[4], randomPeople[5]] = 2

    # for i in range(2, 10, 2):
    #     population[randomPeople[i], randomPeople[i+1]] = 2



    return population


# Update the state
def updatePopulation(population, p, q, r):

    # Create new state
    newState = copy.deepcopy(population)
    
    # Fetch side length of lattice
    N = len(population)

    # Loop over whole population
    for i, row in enumerate(population):
        for j, person in enumerate(row):

            # Resting rule
            if person == resting and random.random() <= p:
                newState[i,j] = sharer
                # print(f"Person is sharer next state")
                continue
            
            # Sharing rule
            if person == sharer and random.random() <= q:

                # Fetch the neighbors
                neighbors = []
                for r in range(-1, 2):
                    for c in range(-1, 2):
                        if r == 0 and c == 0:
                            continue
                        rr = r + i if r + i < N else 0
                        cc = c + j if c + j < N else 0
                        neighbors.append((rr,cc))

                # Pick random neighbor
                randomNeighbour = random.choice(neighbors)

                # If that person is resting then they will now become a sharer
                if population[randomNeighbour[0], randomNeighbour[1]] == resting:
                    newState[randomNeighbour[0],randomNeighbour[1]] = sharer
                    # print(f"{randomNeighbour[0], randomNeighbour[1]} is sharer next state")
                    continue

                # However, if the person they pick is bored, then the sharer will lose interest and become bored too
                elif population[randomNeighbour[0], randomNeighbour[1]] == bored:
                    newState[i,j] = bored
                    # print(f"{i,j} will be bored next iteration")
                    continue

            # Bored rule
            if person == bored and random.random() <= r:

                # Fetch the neighbors
                neighbors = []
                for r in range(-1, 2):
                    for c in range(-1, 2):
                        if r == 0 and c == 0:
                            continue
                        rr = r + i if r + i < N else 0
                        cc = c + j if c + j < N else 0
                        neighbors.append((rr,cc))

                # Pick random neighbor
                randomNeighbour = random.choice(neighbors)

                #  If that person is resting then the bored person will now become resting
                if population[randomNeighbour[0], randomNeighbour[1]] == resting:
                    newState[i,j] = resting
                    # print(f"{i,j} will be resting next iteration")
                    continue

                # Otherwise they will continue to be bored
                else:
                    newState[i,j] = bored
                    # print(f"{i,j} will continue to bored next iteration")
                    continue

    return newState

# Plotting

# Color maps
cmap = ListedColormap(["white", "green", "red"])
resting_patch = mPatches.Patch(color="white", label="Resting")
sharing_patch = mPatches.Patch(color="green", label="Sharing")
boring_patch = mPatches.Patch(color="red", label="Boring")

# Function for plotting
def plotPopulation(population, title):
    plt.figure(1)
    plt.title(title)
    plt.legend(handles=[boring_patch, sharing_patch, resting_patch], loc="lower left")
    plt.imshow(population, vmin=0, vmax=len(cmap.colors), cmap=cmap)
    plt.yticks(color="w")
    plt.show()


def calculateNumberOfSharersLattice(population):
    numberOfSharers = 0
    for i in range(len(population)):
        for j in range(len(population)):
            if population[i,j] == sharer:
                numberOfSharers += 1
    return numberOfSharers

def calculateNumberOfRestingLattice(population):
    numberOfResting = 0
    for i in range(len(population)):
        for j in range(len(population)):
            if population[i,j] == resting:
                numberOfResting += 1
    return numberOfResting

def calculateNumberOfBoredLattice(population):
    numberOfBored = 0
    for i in range(len(population)):
        for j in range(len(population)):
            if population[i,j] == bored:
                numberOfBored += 1
    return numberOfBored

    

if __name__ == "__main__":

    # Initialize population
    N = 10
    population = initialiazePopulation(N)
    # plotPopulation(population, f"Initial state")



    # Set simulation parameters:
    p = 0.001
    q = 0.5
    r = 0.01
    numberOfSimulations = 10
    numberOfSteps = 1000

    numberOfSharersMatrix = np.zeros((numberOfSimulations, numberOfSteps))


    plt.ion()
    # Update population
    for j in range(numberOfSteps):
        population = updatePopulation(population, p, q, r)
        plt.close()
        plotPopulation(population, f"Population at time step {j+1}")
        plt.show()
        plt.pause(0.01)

        
