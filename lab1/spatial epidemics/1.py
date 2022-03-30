

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mPatches
from spatial_epidemics import *


# Define grid size
N = 100

# Initialize population
population = initializePopulation(N)


# Color maps
cmap = ListedColormap(["white", "red"])
susceptible_patch = mPatches.Patch(color="white", label="Susceptible")
infected_patch = mPatches.Patch(color="red", label="Infected")

# Function to plot population for different gammas at different iterations
def plotPopulationGamma(population, title, gamma, iteration):
    plt.figure(1)
    plt.title(title)
    plt.legend(handles=[infected_patch, susceptible_patch], loc="lower left")
    plt.imshow(population, vmin=0, vmax=len(cmap.colors), cmap=cmap)
    plt.yticks(color="w")
    plt.show()
    plt.savefig(f"{gamma}_{iteration}.png")

# gamma = 0.3
# plt.ion()
# for i in range(100):
#     population = updateState(population, gamma, N)
#     if i in [10,30,60,90]:
#         plotPopulationGamma(population, f"Gamma: {gamma}. Iteration: {i}", gamma, i)



# Create gamma dictionary
gamma = {}
nmrGammas = 10
for n in range(nmrGammas):
    gamma[n/nmrGammas] = 0

# Define number of simulations and steps for each simulation
nmrSimulation = 100
nmrSteps = 100

for g, survivalCount in gamma.items():

    for i in range(nmrSimulation):
        # Create population
        population = initializePopulation(N)
        for j in range(nmrSteps):
            # Update population
            population = updateState(population, g, N)
        # If population has any infected person, count it is as survived
        if np.any(population[0,:]):
            survivalCount +=1
    print(f"Gamma: {g}. Survival percentage: {survivalCount/(nmrSimulation)}")
    # Add the percentage that it survived
    gamma[g] = survivalCount/(nmrSimulation)


plt.plot(gamma.keys(), gamma.values())
plt.xlabel("Gamma")
plt.ylabel("Probability that virus survives")
plt.title("Probability that virus survives as a function of gamma")
plt.savefig("gammaSurvives.png")
plt.show()


