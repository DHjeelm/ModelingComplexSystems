from spatial_epidemics import *
import numpy as np
from random import random


def initRandomPopulation(N: int, p: float) -> np.array:
    pop = np.zeros((1, N))

    for i in range(N):
        if random() <= p:
            pop[0, i] = 1
    
    return pop




if __name__ == "__main__":
    N = 100
    nSimulations = 100


    stepSizeGammaP = 100
    nSteps = 100
    survivalCounts = np.zeros(1, stepSizeGammaP**2)

    gammaPSurface = np.zeros(3, stepSizeGammaP)
    gammas = np.linspace(0, 1, stepSizeGammaP)
    ps = np.linspace(0, 1, stepSizeGammaP)

    gammaPSurface[0, :] = gammas
    gammaPSurface[1, :] = ps




    for i in range(stepSizeGammaP):
        # Change gamma

        g = gammas[i]
        for j in range(stepSizeGammaP):
            # Change p
            p = ps[j]
            survivalCount = 0
            
            for k in range(nSimulations):
                population = initRandomPopulation(N, p)
                for h in range(nSteps):
                    population = updateState(population, g)

                if np.any(population[0, :]):
                    survivalCount += 1

            gammaPSurface[3, ] = survivalCount/nSimulations

                
        








