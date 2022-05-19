# File for runnning a simulation of the model. The setup of the file is inspired by: git@github.com:fskerman/vicsek_model.git

# Imports
from cmath import isfinite
import math
# import cv2, os, sys
import os
from turtle import update
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import diff
# from cv2 import VideoWriter, VideoWriter_fourcc
from sympy import re
from geometry import *
from neighbor import *
from measures import *
import pylab as pl


def predatorMovement(particles, thetas, rPredator, rEat, etaPredator, x, y, i, size, predSpeed, timeStep, predators):

    # Get neighbors distance and indices
    neighbors = getClosestNeighbor(particles, rPredator, x, y, i, size)

    # Get neighboring particles indices for current particle
    nearbyPredators, nearbyPrey = getNeighbors(particles, rPredator, x, y, size, predators, i)

    # Sort the neighbors in ascending distance order
    sortedNeighbors = {k: v for k, v in sorted(neighbors.items(), key=lambda item: item[1])}

    minDistance = float("inf")
    for key, value in sortedNeighbors.items():
        # If the neighbor is a predtor, continue
        if key in predators:
            continue
        # If the neighbor is prey, save its distance and index
        else:
            minIndex = key
            minDistance = value
            break

    
    #  Rules for eating
    eaten = 0
    # Eat prey if it is inside rEat
    if minDistance < rEat:
        # Update number of eaten
        eaten = 1
        
        # Prey is teleported to random position with a random angle.
        particles[minIndex,:] = np.random.uniform(0, size)
        thetas[minIndex] = randomAngle()

        # Fetch random angle
        n_angle = randomAngle()

        # Multiply with eta
        noise = etaPredator * n_angle

        # Update theta
        thetas[i] += noise

        # Move to new position
        particles[i,:] += timeStep * predSpeed * angleToVector(thetas[i])

    # Else hunt the closest particle 
    elif minDistance <= rPredator:

        # Find neighbor coords
        neighborX, neighborY = particles[minIndex,:]

        # Find predator coords
        predX, predY = particles[i,:]

        # Find direction to move in using predator sense:
        moveX, moveY = predatorSense(neighborX, neighborY, predX, predY, size)

        # Update the theta
        angle = vectorToAngle((moveX,moveY))

        # Fetch random angle
        n_angle = randomAngle()

        # Multiply with eta
        noise = etaPredator * n_angle

        thetas[i] = angle + noise

        # Update predator position
        particles[i,:] += timeStep * predSpeed * angleToVector(thetas[i])
    
    elif nearbyPredators:

        ##### Alignment ######
        # Get average theta angle
        avgAngle = getAverage(thetas, nearbyPredators)
     
        ####### Cohesion #######
        
        # Get average position of all nearby predators
        meanX, meanY = getAveragePosition(particles[len(particles)-len(predators):len(particles)], rPredator, x, y, i, size)
        
        # Coordinates pointing to mean
        moveX, moveY = moveToMean(meanX, meanY, x, y, size)

        # Calculate angle from coordinates
        cohesionAngle = vectorToAngle((moveX,moveY))

        ######## Seperation ##########
        diff = 0
        for k in nearbyPredators:
            # print(k)
            if k != i:
                diff += (particles[i,:] - particles[k,:])/torusDistance(particles[i,0], particles[i,1], particles[k,0], particles[k,1],size)
        
   
        diffAngle = vectorToAngle((diff[0],diff[1]))

        # Fetch random angle
        n_angle = randomAngle()

        # Multiply with eta
        noise = etaPredator * n_angle

        # Update angle:
        updateAngle = (cohesionAngle-thetas[i])*-0.2 + (avgAngle-thetas[i])*-0.2 + (diffAngle-thetas[i])*-0.1 + noise

        thetas[i] += updateAngle


        # Update position
        particles[i,:] += timeStep * predSpeed* angleToVector(thetas[i])
        
    
    # If no prey inside rPredator to hunt or predators to flock with, move randomly
    else: 

        # print("I can't find no neighbors, lets move randomly")

        # Fetch random angle
        n_angle = randomAngle()

        # Multiply with eta
        noise = etaPredator * n_angle

        # Update theta
        thetas[i] += noise

        # Move to new position
        particles[i,:] += timeStep * predSpeed *  angleToVector(thetas[i])
    
    


    return eaten


def preyMovement(particles, thetas, etaPrey, rPrey, x, y, i, numberOfPredators, size, preySpeed, timeStep, predators):

    # Get neighboring particles indices for current particle
    nearbyPredators, nearbyPrey = getNeighbors(particles, rPrey, x, y, size, predators, i)

    # print(f"Nearby predators: {nearbyPredators}")
    # print(f"Nearby prey: {nearbyPrey}")
    # print()

    if nearbyPredators:

        minIndex = min(nearbyPredators, key=nearbyPredators.get)
        # Find neighbor coords
        neighborX, neighborY = particles[minIndex,:]

        # Find predator coords
        predX, predY = particles[i,:]

        # Find direction that predator sense wants to move in:
        moveX, moveY = predatorSense(neighborX, neighborY, predX, predY, size)

        # Update the theta to move in the opposite direction of the predator movement
        angle = vectorToAngle((moveX,moveY)) + math.pi

        # Fetch random angle
        n_angle = randomAngle()

        # Multiply with eta
        noise = etaPrey * n_angle

        # Update angle:
        updateAngle = (angle-thetas[i])*0.2

        thetas[i] += updateAngle 

        # Update predator position
        particles[i,:] += timeStep * preySpeed * angleToVector(thetas[i])

    elif nearbyPrey:

        ##### Alignment ######
        # Get average theta angle
        avgAngle = getAverage(thetas, nearbyPrey)
     
        ####### Cohesion #######
        
        # Get average position of all nearby predators
        meanX, meanY = getAveragePosition(particles[0:len(particles)-len(predators)], rPrey, x, y, i, size)
        
        # Coordinates pointing to mean
        moveX, moveY = moveToMean(meanX, meanY, x, y, size)

        # Calculate angle from coordinates
        cohesionAngle = vectorToAngle((moveX,moveY))

        ######## Seperation ##########
        diff = 0
        for k in nearbyPrey:
            # print(k)
            if k != i:
                diff += (particles[i,:] - particles[k,:])/torusDistance(particles[i,0], particles[i,1], particles[k,0], particles[k,1],size)
        
   
        diffAngle = vectorToAngle((diff[0],diff[1]))

        # Fetch random angle
        n_angle = randomAngle()

        # Multiply with eta
        noise = etaPredator * n_angle

        # Update angle:
        updateAngle = (cohesionAngle-thetas[i])*0.2 + (avgAngle-thetas[i])*0.2 + (diffAngle-thetas[i])*0.1 + noise

        thetas[i] += updateAngle


        # Update position
        particles[i,:] += timeStep * preySpeed * angleToVector(thetas[i])
    
    # If no predators to escape from or prey to flock with, move randomly
    else:

        # Get average theta angle
        avg = getAverage(thetas, nearbyPrey)

        # Get noise angle
        n_angle = randomAngle()
        noise = etaPrey * n_angle

        # Update theta
        thetas[i] = avg + noise

        # Move to new position
        particles[i,:] += timeStep * preySpeed * angleToVector(thetas[i])

def simulateModel(numberOfPrey, numberOfPredators, etaPrey, etaPredator, rPrey, rPredator, rEat, timeStep, endTime, size, preySpeed, predSpeed, predators):

    # Calculate number of particles
    numberOfParticles = numberOfPrey + numberOfPredators


    # # Debug setting
    # particles = np.zeros((numberOfParticles, 2))
    # thetas = np.zeros((numberOfParticles, 1))
    # # # Prey
    # particles[0,0] += 0.1
    # particles[0,1] += 0.5
    # particles[1,0] += 0.1
    # particles[1,1] += 0.45
    # particles[2,0] += 0.1
    # particles[2,1] += 0.4
    # thetas[0,0] = math.pi
    # thetas[1,0] = math.pi
    # thetas[2,0] = math.pi

    # particles[3,0] += 0.6
    # particles[3,1] += 0.5
    # particles[4,0] += 0.6
    # particles[4,1] += 0.45
    # particles[5,0] += 0.6
    # particles[5,1] += 0.4
    # # thetas[0,0] = math.pi*3/2
    # # thetas[1,0] = math.pi*3/2
    # # thetas[2,0] = math.pi*3/2
    # thetas[4,0] = math.pi
    # thetas[3,0] = math.pi
    # thetas[5,0] = math.pi

    # # Predator
    # particles[3,0] += 0.9
    # particles[3,1] += 0.4
    # thetas[3,0] = math.pi*3/2
    # particles[4,0] += 0.8
    # particles[4,1] += 0.6
    # thetas[4,0] = math.pi

    # particles[0,0] += 0.1
    # particles[0,1] += 0.5
    # thetas[0,0] = 0

    # particles[1,0] += 0.8
    # particles[1,1] += 0.5
    # thetas[1,0] = math.pi
    



    # Real simulation setting

    # Create particles
    particles = np.random.uniform(0, size, size=(numberOfParticles, 2))

    # Initialize random angles
    thetas = np.zeros((numberOfParticles, 1))
    for i, theta in enumerate(thetas):
        thetas[i, 0] = randomAngle()


    polarisationList = []
    eatenList = []
    numberOfEaten = 0


    # print("Creating particle files", end='', flush=True)
    # Start the simulation
    t = 0

    while t < endTime:

        # print(end='.', flush=True)

        # Save coordinates & corresponding thetas to a text file
        simulation = np.concatenate((particles, thetas), axis=1)
        np.savetxt("%.2f.txt" % t, simulation)

        # Update the model
        for i, (x, y) in enumerate(particles):
            
            # Predator
            if i >= numberOfParticles-numberOfPredators:
                # print(f"{i}")
                # print(f"At time step: {round(t*100,2)}")
                eaten = predatorMovement(particles, thetas, rPredator, rEat, etaPredator, x, y, i, size, predSpeed, timeStep, predators)
                numberOfEaten += eaten
                # print()
            # Prey
            else:
                preyMovement(particles, thetas, etaPrey, rPrey, x, y, i, numberOfPredators, size, preySpeed, timeStep, predators)


            # Assure correct boundaries
            if particles[i, 0] < 0:
                particles[i, 0] = size + particles[i, 0]

            if particles[i, 0] > size:
                particles[i, 0] = particles[i, 0] - size

            if particles[i, 1] < 0:
                particles[i, 1] = size + particles[i, 1]

            if particles[i, 1] > size:
                particles[i, 1] = particles[i, 1] - size


        # Update number of eaten
        eatenList.append(numberOfEaten)

        # Remove list within list to calculate polarization
        calcThetas = [item for sublist in thetas for item in sublist]
        # print(calcThetas)
        polarisationList.append(calculatePolarisation(calcThetas, numberOfPrey + numberOfPredators))

        # Update time
        t += timeStep
    print()
    return polarisationList, eatenList

def plotModel(coords, thetas, numberOfPrey, numberOfPredators, etaPrey, etaPredator, rPrey, rPredator, rEat):
    '''Function creating a plot for the current state of the model'''

    # Create color scheme for model
    numberOfParticles = len(coords)

    # Loop through each particle
    for i, (x, y) in enumerate(coords):

        # Determine if predator or not
        if i >= numberOfParticles-numberOfPredators:
            c = "r"
        else:
            c = "g"



        # Plot a particle
        plt.scatter(x, y, color = c, marker = ".")

        # Plot the tail of the particle
        theta = thetas[i]
        v = angleToVector(theta)
        x1 = x - (0.025 * v[0])
        y1 = y - (0.025 * v[1])
        plt.plot([x, x1], [y, y1], color=c)

        # textstr = f"Parameters:\n\nPrey:\nNumber of prey: {numberOfPrey}\netaPrey: {etaPrey}\nrPrey: {rPrey}\n\nPredator\nNumber of predators: {numberOfPredators}\netaPredator: {etaPredator}\nrPredator: {rPredator}\nrEat: {rEat}"
        # plt.text(0.02, 0.5, textstr, fontsize=10, transform=plt.gcf().transFigure)
        # plt.subplots_adjust(left=0.3)

    return

def savePlot(path, fname, etaPrey, etaPredator, size, i):
    '''Function saving a plot for the current state of the model'''
    # Axes between 0 and size
    plt.axis([0, size, 0, size])

    # Remove tick marks
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])

    # Title
    plt.title(f"Simulation")

    # Save plot
    plt.savefig(os.path.join(path, fname[:-4]+".jpg"))
    plt.close()

    # Clear for next plot
    plt.cla()

    return
def createPlots(particleDir, numberOfPredators, size):
    '''Function that create plots from the txt.files of coordinates and thetas'''

    print("Creating plots", end='', flush=True)

    # Read text files
    txtFiles = [i for i in os.listdir(particleDir) if i.endswith(".txt")]

    for i, fname in enumerate(txtFiles):
        print(end = ".", flush=True)

        # Fetch file
        f = os.path.join(particleDir, fname)

        # Read in from file
        data = np.loadtxt(f)
        coords = data[:,0:2]
        thetas = data[:,2]

        # Plot the current state and save it
        plotModel(coords, thetas, numberOfPredators)
        savePlot(plotDir, fname, etaPrey, etaPredator, size, i)
    print()

def makeVideo(plotDir):
    """Function making a video of the simulation"""

    # Fetch the plots
    jpgFiles = sorted([i for i in os.listdir(plotDir) if i.endswith("jpg")])

    # Array for all images
    imageArray = []

    # Setting for VideoWriter
    one_size = [0,0]

    for filename in jpgFiles:
        # Fecth file
        f = os.path.join(plotDir, filename)
        # Read as imag
        img = cv2.imread(f)

        # Fetch attributes
        height, width, layers = img.shape
        size = (width,height)
        one_size = [width, height]

        # Add to array
        imageArray.append(img)

    # Create path for video
    video_path = os.path.join(simulationDir, "simulation.mp4")
    print("Saving video as", video_path)

    # Create video
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, tuple(one_size))
    for i in range(len(imageArray)):
        out.write(imageArray[i])
    out.release()


def plotModelWithoutSaving(particleDir, size, numberOfPrey, numberOfPredators, etaPrey, etaPredator, rPrey, rPredator, rEat, polarisation, eaten, preySpeed, predSpeed):
    # Read text files
    txtFiles = [i for i in os.listdir(particleDir) if i.endswith(".txt")]

    # Sort the files
    sortedFiles = sorted(txtFiles)

    # plt.ion()
    for i, fname in enumerate(sortedFiles):
        print(end = ".", flush=True)

        # Fetch file
        f = os.path.join(particleDir, fname)

        # Read in from file
        data = np.loadtxt(f)
        coords = data[:,0:2]
        thetas = data[:,2]

        # Plot the current state
        # plt.close()

        # Set axes between 0 and 1
        plt.axis([0, size, 0, size])

        # Remove tick marks
        frame = plt.gca()
        frame.axes.get_xaxis().set_ticks([])
        frame.axes.get_yaxis().set_ticks([])
        

        # Plot
        plotModel(coords, thetas, numberOfPrey, numberOfPredators, etaPrey, etaPredator, rPrey, rPredator, rEat)
        plt.title(f"Simulation at time step {i}")
        textstr = f"Parameters:\nNumber of prey: {numberOfPrey}\netaPrey: {etaPrey}\nrPrey: {rPrey}\nPrey speed: {preySpeed}\nNumber of predators: {numberOfPredators}\netaPredator: {etaPredator}\nrPredator: {rPredator}\nrEat: {rEat}\nPredator speed: {predSpeed}\n\n\n\nMeasurements:\nNumber of eaten prey: {eaten[i]}"
        plt.text(0.02, 0.3, textstr, fontsize=10, transform=plt.gcf().transFigure)
        plt.subplots_adjust(left=0.3)

        plt.show()
        # plt.pause(0.01)



# Simulation
if __name__ == '__main__':

    # Directory creation
    simulationDir = os.path.join(os.getcwd(), "Simulation")
    particleDir = os.path.join(simulationDir, "Particles")
    plotDir = os.path.join(simulationDir, "Plots")
    if not os.path.exists(simulationDir):
        os.mkdir(simulationDir)
    if not os.path.exists(particleDir):
        os.mkdir(os.path.join(simulationDir, "Particles"))
    if not os.path.exists(plotDir):
        os.mkdir(os.path.join(simulationDir, "Plots"))

    # Change directory to the particle dir
    os.chdir(particleDir)

    ########## Simulation parameters  ############

    # Size of board
    size = 1

    # Number of preys
    numberOfPrey = 40

    # Number of predators
    numberOfPredators = 10

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
    predSpeed = 1.5

    # Time settings
    t = 0.0
    timeStep = 0.01
    endTime = 2

    # Find indices of all predators
    predators = [i for i in range(numberOfPrey, numberOfPrey+numberOfPredators)]


    # Simulate model
    polarisation, eaten = simulateModel(numberOfPrey, numberOfPredators, etaPrey, etaPredator, rPrey, rPredator, rEat, timeStep, endTime, size, preySpeed, predSpeed, predators)
    # print(eaten)
    plotModelWithoutSaving(particleDir, size, numberOfPrey, numberOfPredators, etaPrey, etaPredator, rPrey, rPredator, rEat, polarisation, eaten, preySpeed, predSpeed)
    # print(numberOfEaten)

    # # print(polarisation)

    # plt.plot(list(range(len(polarisation))), polarisation)
    # plt.show()


    # # Create plots
    # createPlots(particleDir, numberOfPredators, size)

    # # # Make video
    # makeVideo(plotDir)

    # print(polarisation)




