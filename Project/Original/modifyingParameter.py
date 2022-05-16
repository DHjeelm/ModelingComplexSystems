
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

def modifyingPredatorSpeed():
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
    rPrey = 0.2
    rPredator = 0.2

    # Eat radius
    rEat = 0.05

    # Speed
    preySpeed = 1

    # Time settings
    timeStep = 0.01
    endTime = 1

    predSpeedVector = np.linspace(0.1,10,20)
    numberOfSimulations = 50
    numberOfEatenList = np.linspace(0, 1, numberOfSimulations*predSpeedVector.size)

    for count, speed in enumerate(predSpeedVector):
        
        predSpeed = speed
        for sim in range(numberOfSimulations):
            # Simulate model
            polarisation, eaten = simulateModel(numberOfPrey, numberOfPredators, etaPrey, etaPredator, rPrey, rPredator, rEat, timeStep, endTime, size, preySpeed, predSpeed)
            print(f"Number of eaten at {sim} with predSpeed = {predSpeed} is: { eaten[-1]}")
            numberOfEatenList[count*numberOfSimulations+sim] = eaten[-1]

    predSpeedVector = predSpeedVector.repeat(numberOfSimulations)
    plt.figure(1)
    plt.hist2d(predSpeedVector, numberOfEatenList, bins=10)
    plt.xlabel("Predator speed")
    plt.ylabel(f"Number of eaten prey")
    plt.title("Phase transition plot of number of eaten prey vs predator speed")
    plt.colorbar()
    plt.show()

def modifyingRPredator():
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
    rPrey = 0.2

    # Eat radius
    rEat = 0.05

    # Speed
    preySpeed = 1
    predSpeed = 1.5

    # Time settings
    timeStep = 0.01
    endTime = 1

    rPredatorVector = np.linspace(0.01,np.sqrt(size/2),20)
    numberOfSimulations = 50
    numberOfEatenList = np.linspace(0, 1, numberOfSimulations*rPredatorVector.size)

    for count, rPred in enumerate(rPredatorVector):
        
        rPredator = rPred
        for sim in range(numberOfSimulations):
            # Simulate model
            polarisation, eaten = simulateModel(numberOfPrey, numberOfPredators, etaPrey, etaPredator, rPrey, rPredator, rEat, timeStep, endTime, size, preySpeed, predSpeed)
            print(f"Number of eaten at {sim} with rPredator = {rPredator} is: { eaten[-1]}")
            numberOfEatenList[count*numberOfSimulations+sim] = eaten[-1]

    rPredatorVector = rPredatorVector.repeat(numberOfSimulations)
    plt.figure(2)
    plt.hist2d(rPredatorVector, numberOfEatenList, bins=10)
    plt.xlabel("rPredator")
    plt.ylabel(f"Number of eaten prey")
    plt.title("Phase transition plot of number of eaten prey vs rPredator")
    plt.colorbar()
    plt.show()

def modifyingEtaPrey():
    ########## Simulation parameters  ############

    # Size of board
    size = 1

    # Number of preys
    numberOfPrey = 50

    # Number of predators
    numberOfPredators = 1

    # Eta (randomness factor)
    etaPredator = 0.2

    # Visual radius for prey and predator
    rPrey = 0.2
    rPredator = 0.2

    # Eat radius
    rEat = 0.05

    # Speed
    preySpeed = 1
    predSpeed = 1.5

    # Time settings
    timeStep = 0.01
    endTime = 1

    etaPreyVector = np.linspace(0,1,20)
    numberOfSimulations = 50
    numberOfEatenList = np.linspace(0, 1, numberOfSimulations*etaPreyVector.size)

    for count, eta in enumerate(etaPreyVector):
        
        etaPrey = eta
        for sim in range(numberOfSimulations):
            # Simulate model
            polarisation, eaten = simulateModel(numberOfPrey, numberOfPredators, etaPrey, etaPredator, rPrey, rPredator, rEat, timeStep, endTime, size, preySpeed, predSpeed)
            print(f"Number of eaten at {sim} with etaPrey = {etaPrey} is: { eaten[-1]}")
            numberOfEatenList[count*numberOfSimulations+sim] = eaten[-1]

    etaPreyVector = etaPreyVector.repeat(numberOfSimulations)
    plt.figure(3)
    plt.hist2d(etaPreyVector, numberOfEatenList, bins=10)
    plt.xlabel("Eta prey")
    plt.ylabel(f"Number of eaten prey")
    plt.title("Phase transition plot of number of eaten prey vs eta prey")
    plt.colorbar()
    plt.show()

def modifyingEtaPredator():
    ########## Simulation parameters  ############

    # Size of board
    size = 1

    # Number of preys
    numberOfPrey = 50

    # Number of predators
    numberOfPredators = 1

    # Eta (randomness factor)
    etaPrey = 0.2

    # Visual radius for prey and predator
    rPrey = 0.2
    rPredator = 0.2

    # Eat radius
    rEat = 0.05

    # Speed
    preySpeed = 1
    predSpeed = 1.5

    # Time settings
    timeStep = 0.01
    endTime = 1

    etaPredVector = np.linspace(0,1,20)
    numberOfSimulations = 50
    numberOfEatenList = np.linspace(0, 1, numberOfSimulations*etaPredVector.size)

    for count, eta in enumerate(etaPredVector):
        
        etaPredator = eta
        for sim in range(numberOfSimulations):
            # Simulate model
            polarisation, eaten = simulateModel(numberOfPrey, numberOfPredators, etaPrey, etaPredator, rPrey, rPredator, rEat, timeStep, endTime, size, preySpeed, predSpeed)
            print(f"Number of eaten at {sim} with etaPred = {etaPredator} is: { eaten[-1]}")
            numberOfEatenList[count*numberOfSimulations+sim] = eaten[-1]

    etaPredVector = etaPredVector.repeat(numberOfSimulations)
    plt.figure(4)
    plt.hist2d(etaPredVector, numberOfEatenList, bins=10)
    plt.xlabel("Eta predator")
    plt.ylabel(f"Number of eaten prey")
    plt.title("Phase transition plot of number of eaten prey vs eta predator")
    plt.colorbar()
    plt.show()


if __name__ == '__main__':

    # modifyingRPredator()
    # modifyingPredatorSpeed()
    # modifyingEtaPrey()
    modifyingEtaPredator()

