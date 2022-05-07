# File for runnning a simulation of the model. The setup of the file is inspired by: git@github.com:fskerman/vicsek_model.git

# Imports
import math
import cv2, os, sys
import numpy as np
import matplotlib.pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc
from geometry import *
from neighbor import *


def predatorMovement(particles, thetas, rPredator, rEat, eta, x, y, i, size, t):

    # Get closest neighbors distance and indices
    minIndex, minDistance = getClosestNeighbor(particles, rPredator, x, y, i, size)

    # # Rules for eating
    # numberOfEaten = 0
    if minDistance < rEat:
        
        # Prey is teleported to random position with a random angle.
        particles[minIndex,:] = np.random.uniform(0, size)
        thetas[minIndex] = randomAngle()
        # print(f"At time step {round(t*100)} predator ate prey {minIndex}")

        # Fetch random angle
        n_angle = randomAngle()

        # Multiply with eta
        noise = eta * n_angle

        # Update theta
        thetas[i] += noise

        # Move to new position
        particles[i,:] += timeStep * angleToVector(thetas[i])

    # If none inside rPredator move randomly
    elif minDistance >= rPredator:

        # print("I can't find no neighbors, lets move randomly")


        # Fetch random angle
        n_angle = randomAngle()

        # Multiply with eta
        noise = eta * n_angle

        # Update theta
        thetas[i] += noise

        # Move to new position
        particles[i,:] += timeStep * angleToVector(thetas[i])



    else:

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
        noise = eta * n_angle

        thetas[i] = angle + noise

        # Update predator position
        particles[i,:] += timeStep * angleToVector(thetas[i])


        # Previous move rules

        # Find angle
        # distance, angle = torusDistance(neighborX, neighborY, predX, predY, size)

        # # Find difference between neighbor and predator
        # norm = torusDistance(neighborX, neighborY, predX, predY)
        # diffX = (neighborX - predX)/norm
        # diffY = (neighborY - predY)/norm

        # # Calculate update angle
        # phi = atan2(diffY, diffX) + np.random.uniform(0, eta/3) * math.pi

        # if angle < 0:
        #     phi += 2*math.pi

        # print(f"At time step {np.round(t*100)}: Neighbor is at {(neighborX, neighborY)}, i am at {predX, predY}")
        # print(f"I want to move {moveX, moveY} and the angle is {angle}")
        # print()


def preyMovement(particles, thetas, eta, rPrey, x, y, i, numberOfPredators, size):

    # Get neighboring prey indices for current particle
    neighbors = getNeighbors(particles[:-numberOfPredators], rPrey, x, y, size)

    # # Debug
    # avg = -math.pi*3/4

    # Get average theta angle
    avg = getAverage(thetas, neighbors)

    # Get noise angle
    n_angle = randomAngle()
    noise = eta * n_angle

    # Update theta
    thetas[i] = avg + noise

    # Move to new position
    particles[i,:] += timeStep * angleToVector(thetas[i])

def simulateModel(numberOfPrey, numberOfPredators, etaPrey, etaPredator, rPrey, rPredator, rEat, timeStep, endTime, size):

    # Calculate number of particles
    numberOfParticles = numberOfPrey + numberOfPredators


    # Debug setting
    # particles = np.zeros((numberOfParticles, 2))
    # thetas = np.zeros((numberOfParticles, 1))
    # # # Prey
    # particles[0,0] += 0.1
    # particles[0,1] += 0.5
    # particles[1,0] += 0.1
    # particles[1,1] += 0.45
    # particles[2,0] += 0.1
    # particles[2,1] += 0.4
    # # thetas[0,0] = math.pi*3/2
    # # thetas[1,0] = math.pi*3/2
    # # thetas[2,0] = math.pi*3/2
    # # thetas[0,0] = 0
    # # thetas[1,0] = 0
    # # thetas[2,0] = 0

    # # Predator
    # particles[3,0] += 0.9
    # particles[3,1] += 0.5
    # thetas[3,0] = math.pi


    # Real simulation setting

    # Create particles
    particles = np.random.uniform(0, size, size=(numberOfParticles, 2))

    # Initialize random angles
    thetas = np.zeros((numberOfParticles, 1))
    for i, theta in enumerate(thetas):
        thetas[i, 0] = randomAngle()


    polarisationList = []

    print("Creating particle files", end='', flush=True)
    # Start the simulation
    t = 0
    while t < endTime:

        print(end='.', flush=True)

        # Save coordinates & corresponding thetas to a text file
        simulation = np.concatenate((particles, thetas), axis=1)
        np.savetxt("%.2f.txt" % t, simulation)

        # Update the model
        for i, (x, y) in enumerate(particles):

            # Predator
            if i >= numberOfParticles-numberOfPredators:
                predatorMovement(particles, thetas, rPredator, rEat, etaPredator, x, y, i, size, t)
            # Prey
            else:
                preyMovement(particles, thetas, etaPrey, rPrey, x, y, i, numberOfPredators, size)


            # Assure correct boundaries
            if particles[i, 0] < 0:
                particles[i, 0] = size + particles[i, 0]

            if particles[i, 0] > size:
                particles[i, 0] = particles[i, 0] - size

            if particles[i, 1] < 0:
                particles[i, 1] = size + particles[i, 1]

            if particles[i, 1] > size:
                particles[i, 1] = particles[i, 1] - size



        # Remove list within list to calculate polarization
        calcThetas = [item for sublist in thetas for item in sublist]
        # print(calcThetas)
        polarisationList.append(calculatePolarisation(calcThetas, numberOfPrey + numberOfPredators))

        # Update time
        t += timeStep
    print()
    return polarisationList

def plotModel(coords, thetas, numberOfPredators):
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

    return

def savePlot(path, fname, etaPrey, etaPredator, size):
    '''Function saving a plot for the current state of the model'''
    # Axes between 0 and 1
    plt.axis([0, size, 0, size])

    # remove tick marks
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])

    # Title
    plt.title(f"Simulation with ηPrey = {etaPrey} and ηPred = {etaPredator}")

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

    for fname in txtFiles:
        print(end = ".", flush=True)

        # Fetch file
        f = os.path.join(particleDir, fname)

        # Read in from file
        data = np.loadtxt(f)
        coords = data[:,0:2]
        thetas = data[:,2]

        # Plot the current state and save it
        plotModel(coords, thetas, numberOfPredators)
        savePlot(plotDir, fname, etaPrey, etaPredator, size)
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

def calculatePolarisation(thetas, numberOfParticles):
    # Calculate cosine and sine sum
    cosSum = (np.sum(np.cos(thetas)))**2
    sinSum = (np.sum(np.sin(thetas)))**2
    # Polarisation
    polarisation = 1/numberOfParticles*np.sqrt(cosSum+ sinSum)

    return polarisation

def plotModelWithoutSaving(size):
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
        plotModel(coords, thetas, numberOfPredators)
        plt.title(f"Simulation at time step {i}")
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
    numberOfPredators = 1

    # Eta (randomness factor)
    etaPrey = 0
    etaPredator = 0.2

    # Visual radius for prey and predator
    rPrey = 0.2
    rPredator = 0.1

    # Eat radius
    rEat = 0.05

    # Time settings
    t = 0.0
    timeStep = 0.01
    endTime = 1


    # Simulate model
    polarisation = simulateModel(numberOfPrey, numberOfPredators, etaPrey, etaPredator, rPrey, rPredator, rEat, timeStep, endTime, size)

    plotModelWithoutSaving(size)

    # # print(polarisation)

    # plt.plot(list(range(len(polarisation))), polarisation)
    # plt.show()


    # # Create plots
    # createPlots(particleDir, numberOfPredators, size)

    # # Make video
    # makeVideo(plotDir)

    # print(polarisation)




