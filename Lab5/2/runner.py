from numpy import argmax, size
from regex import P
from painter_play import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mPatches
import random

def plotChromosome(score, xpos, ypos, room):
    plotRoom = room

    cmap = ListedColormap(["white", "green", "red"])
    notPainted = mPatches.Patch(color="white", label="Not painted")
    paintedOnce = mPatches.Patch(color="green", label="Painted once")
    paintedSeveral = mPatches.Patch(color="red", label="Painted several times")
    for i in range(len(xpos)):
        # If painter have been on position once
        
        if plotRoom[xpos[i]-1,ypos[i]-1] == 0:
            plotRoom[xpos[i]-1,ypos[i]-1] = 1

        # Painter have been on this position more than once
        else:
            plotRoom[xpos[i]-1,ypos[i]-1] = 2
        
        plt.figure(1, figsize=(18, 8))
        plt.legend(handles=[paintedSeveral, paintedOnce,
                            notPainted], loc="best")
        plt.imshow(plotRoom, vmin=0, vmax=len(cmap.colors), cmap=cmap)
        plt.text(ypos[i]-1, xpos[i]-1, str("Painter"), va='center', ha='center', size=5)
        # plt.imshow(plotRoom)
        plt.yticks(color="w")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Chromosome with score {round(score,2)}")

        plt.show()

def findNextGeneration(generation, fitness):

    # Define mutation rate
    mutationRate = 0.002

    # Find index where to do crossover
    crossOverCut = random.randint(0,len(generation[1]))

    # Calculate the cumalative sum of fitness and normalize it
    cumSumFitness = np.cumsum(fitness)/sum(fitness)
    
    # Create next generation
    nextGeneration = np.zeros_like(generation)
    for i in range(len(nextGeneration)):

        # Pick parents randomly based on cumsum
        parents = []
        for p in range(2):
            # Find first element bigger than random 
            idx = next(x[0] for x in enumerate(cumSumFitness) if x[1] > random.random())
            # Add that index to parents
            parents.append(idx)

        # Perform crossover for the new chromosome
        nextChromosome = np.zeros_like(generation[0])
        nextChromosome[0:crossOverCut] = generation[parents[1], 0:crossOverCut]
        nextChromosome[crossOverCut+1:-1] = generation[parents[1], crossOverCut+1:-1]

        # Perform mutation
        for k in range(len(nextChromosome)):
            if mutationRate >= random.random():
                nextChromosome[k] = random.randint(0,3)

        # Add the chromosome to the nextGeneration      
        nextGeneration[i] = nextChromosome

    return nextGeneration

if __name__ == "__main__":

    # # Initialize room and rules
    # test_room=np.zeros((20,40))
    # test_rules=np.ones((54,1))
    # for i in range(len(test_rules)):
    #     test_rules[i]=3

    # Create a empty room of size (20 x 40)
    room=np.zeros((20,40))

    # Create a first gener matrix of 50 chromosomes (50x54)
    lengthOfChromosomes = 54
    numberOfChromosomes = 50
    firstGeneration = np.matrix.round(np.random.uniform(0, 3, size=(numberOfChromosomes, lengthOfChromosomes)))

    # Settings
    numberOfTimesToRunRules = 10
    numberOfGenerations = 100

    # The last generation and its fitness
    lastFitness = []
    lastGeneration = np.zeros((numberOfChromosomes, lengthOfChromosomes))

    generation = firstGeneration
    for i in range(numberOfGenerations):
        print(f"Generation {i}")
        # Create list to save fitness
        fitness = []
        for j in range(numberOfChromosomes):
            innerFitness = 0
            for k in range(numberOfTimesToRunRules):
                score, xpos, ypos = painter_play(firstGeneration[i], room)
                innerFitness+=score
            
            fitness.append(innerFitness/numberOfTimesToRunRules)
        print(np.max(fitness))
        # Save previous generation
        prevGeneration = generation
        generation = findNextGeneration(generation, fitness)
        if i == numberOfGenerations-1:
            maxFitness = np.argmax(fitness)
            bestChromosome = prevGeneration[maxFitness]
            lastFitness = fitness
            lastGeneration = generation

    # print(bestChromosome)
    # print(lastFitness)

    # Run painter
    score, xpos, ypos = painter_play(bestChromosome, room)
    # Plot painter
    plotChromosome(score, xpos, ypos, room)


    



