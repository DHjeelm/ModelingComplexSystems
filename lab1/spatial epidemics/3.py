import time
import numpy as np
import matplotlib.pyplot as plt
from random import random
from copy import deepcopy
from matplotlib.colors import ListedColormap
import matplotlib.patches as mPatches
from sympy import im

# State to number representation
susceptible = 0
infected = 1
recovered = 2
immune = 3

# Time for a recovered person to become susceptible
timeToBecomeSusceptible = 14

# The built-in probabilites of the model
probabilities = {"initiallyInfected": .001,
                 immune: 0.05, "recovery": .5, "infection": .6}


# Function that initiliaze the population with infected individuals
def initPopulation(N: int):
    nrInfected = 0
    population = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if random() <= probabilities["initiallyInfected"]:
                population[i, j] = infected
                nrInfected += 1
    if nrInfected == 0:
        population[N//2, N//2] = infected

    return population

# Function that updates the state of the population according to the rules 
def updatePopulation(population: np.ndarray, recoveredMask: np.ndarray):
    newState = deepcopy(population)
    N = len(population)

    for i, row in enumerate(population):
        for j, person in enumerate(row):

            # Immune rule
            if person == immune:
                continue

            # Recovered rule
            if person == recovered:
                if recoveredMask[i, j] == 0:
                    newState[i, j] = susceptible
                    continue
                else:
                    recoveredMask[i, j] -= 1
                continue

            # Infected rule
            if person == infected and random() <= probabilities["recovery"]:
                if random() <= probabilities[immune]:
                    newState[i, j] = immune
                else:
                    newState[i, j] = recovered
                    recoveredMask[i, j] = timeToBecomeSusceptible
                continue

            # Susceptible rule
            for r in range(-1, 2):
                for c in range(-1, 2):
                    if r == 0 and c == 0:
                        continue
                    rr = r + i if r + i < N else 0
                    cc = c + j if c + j < N else 0
                    if population[rr, cc] == infected and random() <= probabilities["infection"]:
                        newState[i, j] = infected
                        break
                else:
                    continue
                break

    return newState

# Function that plots the population with colors and labels
def plotPopulation(population, title: str, nSteps, iteration, susceptibleCountArray, infectedCountArray, recoveredCountArray, immuneCountArray):

    cmap = ListedColormap(["white", "red", "yellow", "blue"])
    susceptible_patch = mPatches.Patch(color="white", label="Susceptible")
    infected_patch = mPatches.Patch(color="red", label="Infected")
    recovered_patch = mPatches.Patch(color="yellow", label="Recovered")
    immune_patch = mPatches.Patch(color="blue", label="Immune")
    neverInfected_patch = mPatches.Patch(color="green", label="Never Infected")

    # Counting variables
    susceptibleCount = 0
    infectedCount = 0
    recoveredCount = 0
    immuneCount = 0 

    for i, row in enumerate(population):
        for j, person in enumerate(row):
            
            # Count immune
            if person == immune:
                immuneCount +=1
                
            # Count susceptible
            elif person == susceptible:
                susceptibleCount +=1
            
            # Count infected
            elif person == infected:
                infectedCount +=1

            # Count recovered
            else: 
                recoveredCount +=1

    # Fill arrays
    susceptibleCountArray[iteration] = susceptibleCount
    infectedCountArray[iteration] = infectedCount
    recoveredCountArray[iteration] = recoveredCount
    immuneCountArray[iteration] = immuneCount


 
    # Plotting
    plt.figure(1, figsize=(16, 8), dpi=80)
    plt.subplot(121)
    plt.title(title)
    plt.legend(handles=[infected_patch, susceptible_patch,
                        recovered_patch, immune_patch], bbox_to_anchor=(-0.3,0.5), loc="center left", borderaxespad=0)
    plt.imshow(population, vmin=0, vmax=len(cmap.colors), cmap=cmap)
    plt.yticks(color="w")

    plt.subplot(122)
    plt.plot(infectedCountArray, color='red', label='Infected')
    plt.plot(recoveredCountArray, color='yellow', label='Recovered')
    plt.plot(immuneCountArray, color='blue', label='Immune')
    plt.plot(susceptibleCountArray, color='white', label='Susceptible')
    ax = plt.gca()
    ax.set_facecolor('#d3d3d3')
    plt.title(f"Statistics at iteration {iteration+1}")
    plt.xlim([0, nSteps])

    plt.show()



def plotStatistics(population, iteration, nSteps, title, susceptibleCountArray, infectedCountArray, recoveredCountArray, immuneCountArray):

    # Counting variables
    susceptibleCount = 0
    infectedCount = 0
    recoveredCount = 0
    immuneCount = 0 

    for i, row in enumerate(population):
        for j, person in enumerate(row):
            
            # Count immune
            if person == immune:
                immuneCount +=1
                
            # Count susceptible
            elif person == susceptible:
                susceptibleCount +=1
            
            # Count infected
            elif person == infected:
                infectedCount +=1

            # Count recovered
            else: 
                recoveredCount +=1

    # Fill arrays
    susceptibleCountArray[iteration] = susceptibleCount
    infectedCountArray[iteration] = infectedCount
    recoveredCountArray[iteration] = recoveredCount
    immuneCountArray[iteration] = immuneCount


    # Plotting
    plt.plot(infectedCountArray, color='red', label='Infected')
    plt.plot(recoveredCountArray, color='yellow', label='Recovered')
    plt.plot(immuneCountArray, color='blue', label='Immune')
    plt.plot(susceptibleCountArray, color='grey', label='Susceptible')
    plt.title(title)
    plt.xlim([0, nSteps])


# Simulation
if __name__ == "__main__":

    # Size of board
    N = 100

    # List that takes care of the timeToBecomeSuseptible 
    recoveredMask = np.zeros((N, N))

    # Initilaize population
    population = initPopulation(N)

    # Plot initial state
    # plotPopulation(population, "Initial state")

    # Number of simulation steps
    nSteps = 100

    # Statistics arrays
    susceptibleCountArray = np.zeros(nSteps)
    infectedCountArray = np.zeros(nSteps)
    recoveredCountArray = np.zeros(nSteps)
    immuneCountArray = np.zeros(nSteps)


    plt.ion()
    for i in range(nSteps):
        
        # Update population
        population = updatePopulation(population, recoveredMask)

        # Check if the epidemic is cured
        if not np.any(population == 1):
            plt.close()
            plt.ioff()
            plotPopulation(population, f"Population {i + 1}", nSteps, i, susceptibleCountArray, infectedCountArray, recoveredCountArray, immuneCountArray)
            break

        # Plot population
        plt.close()
        plotPopulation(population, f"Population {i + 1}", nSteps, i, susceptibleCountArray, infectedCountArray, recoveredCountArray, immuneCountArray)
        plt.show()
        plt.pause(0.01)

    plt.close()
    plt.ioff()
    plotPopulation(population, f"Population {i + 1}", nSteps, i, susceptibleCountArray, infectedCountArray, recoveredCountArray, immuneCountArray)
    plt.show()