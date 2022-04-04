from spatial_epidemics import *
import numpy as np
from random import random



# Initialize population where each population has a probability p of being infected
def initRandomPopulation(N: int, p: float) -> np.array:
    pop = np.zeros((1, N))
    for i in range(N):
        if random() <= p:
            pop[0, i] = 1
    
    return pop




if __name__ == "__main__":

    # Define parameters
    N = 100
    nSimulations = 10
    nStepsGammaP = 200
    nSteps = 50


    survivalCounts = np.zeros((1, nStepsGammaP**2))
    gammaPSurface = np.zeros((nStepsGammaP, nStepsGammaP))
    gammas = np.linspace(0, 1, nStepsGammaP)
    ps = np.linspace(0, 1, nStepsGammaP)


    # Loop over gamma
    for i in range(nStepsGammaP):

        g = gammas[i]

        # Loop over p
        for j in range(nStepsGammaP):
            p = ps[j]
            survivalCount = 0
            
            # Simulate
            for k in range(nSimulations):
                population = initRandomPopulation(N, p)
                for h in range(nSteps):
                    population = updateState(population, g, N)

                # Check any survivors after nSteps
                if np.any(population[0, :]):
                    survivalCount += 1

            # Add survivor percentage
            gammaPSurface[i, j] = survivalCount/nSimulations

    print(gammas)
    print(ps)
    print(gammaPSurface)
    gammas, ps = np.meshgrid(gammas, ps, indexing="ij")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(gammas, ps, gammaPSurface)
    plt.xlabel("gamma")
    plt.ylabel("p")

    plt.show()

                
        








