
# Imports
import math
import cv2, os, sys
import numpy as np
import matplotlib.pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc
from sympy import re
from geometry import *
from neighbor import *
from measures import *
from simulation import *
import pylab as pl


def modifyingRPrey():
    ########## Simulation parameters  ############

    # Size of board
    size = 1

    # Number of preys
    numberOfPrey = 50

    # Number of predators
    numberOfPredators = 1

    # Eta (randomness factor)
    etaPrey = 0.2
    etaPredator = 0.2

    # Visual radius for prey and predator
    rPredator = 0.2

    # Eat radius
    rEat = 0.05

    # Speed
    preySpeed = 1
    predSpeed = 1.5

    # Time settings
    timeStep = 0.01
    endTime = 1

    rPreyVector = np.linspace(0,sqrt(size/2),20)
    numberOfSimulations = 50
    numberOfEatenList = np.linspace(0, 1, numberOfSimulations*rPreyVector.size)
    predators = [i for i in range(numberOfPrey, numberOfPrey+numberOfPredators)]

    for count, r in enumerate(rPreyVector):
        
        rPrey = r
        for sim in range(numberOfSimulations):
            # Simulate model
            polarisation, eaten = simulateModel(numberOfPrey, numberOfPredators, etaPrey, etaPredator, rPrey, rPredator, rEat, timeStep, endTime, size, preySpeed, predSpeed, predators)
            print(f"Number of eaten at {sim} with rPrey = {rPrey} is: { eaten[-1]}")
            numberOfEatenList[count*numberOfSimulations+sim] = eaten[-1]

    rPreyVector = rPreyVector.repeat(numberOfSimulations)
    plt.figure(4)
    plt.hist2d(rPreyVector, numberOfEatenList, bins=10)
    plt.xlabel("rPrey")
    plt.ylabel(f"Number of eaten prey")
    plt.title("Phase transition plot of number of eaten prey vs rPrey")
    plt.colorbar()
    plt.show()


if __name__ == '__main__':

    # modifyingRPredator()
    # modifyingPredatorSpeed()
    # modifyingEtaPrey()
    # modifyingEtaPredator()
    modifyingRPrey()

