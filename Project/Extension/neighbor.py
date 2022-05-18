#!/usr/bin/python
import numpy as np
from sklearn import neighbors
from geometry import *

# File with functions needed for neighbor calculation of the model. 
# getNeighbors and getAverage is inspired by: git@github.com:fskerman/vicsek_model.git

def getClosestNeighbor(particles, r, x0, y0, i, size):
    ''' Function returning the index of the closest neighbor'''

    # Remove yourself as a neighbor
    checkNeighbors = [x for l,x in enumerate(particles) if l!=i] 
    neighbors = {}
    for j,(x1,y1) in enumerate(checkNeighbors):

        dist = torusDistance(x0, y0, x1, y1, size)

        if dist < r:
            neighbors[j] = dist
        else:
            neighbors[j] = float('inf')
        
    # # Get minimum value and index
    # minDistance = min(neighbors)
    # minIndex = neighbors.index(minDistance)

    # Return minimum value and index
    return neighbors


def getNeighbors(particles, r, x0, y0, size, predators, i):
    ''' Function returning all indices of nearbyPrey and distance and indice of nearby predators'''

    nearbyPredators = {}
    nearbyPrey = []

    for j,(x1,y1) in enumerate(particles):
        if j!= i:
            dist = torusDistance(x0, y0, x1, y1, size)
            if dist <= r and j in predators:
                nearbyPredators[j] = dist
            elif dist <= r and j not in predators:
                nearbyPrey.append(j)
            else:
                continue
    return nearbyPredators, nearbyPrey


def getAverage(thetas, neighbors):

    ''' Function calculating average unit vectors for all angles. Returns average angle '''
    
    n_neighbors = len(neighbors)
    avgVector = np.zeros(2)

    for index in neighbors:
        theta = thetas[index,0]
        theta_vec = angleToVector(theta)
        avgVector += theta_vec

    avgAngle = vectorToAngle(avgVector)

    return avgAngle

def checkIfAnyNeighbors(particles, r, x0, y0, i, size):
    # Remove yourself from the list
    checkNeighbors = [x for l,x in enumerate(particles) if l!=i]

    # Loop through particles and check if you have any neighbors
    for j,(x1,y1) in enumerate(checkNeighbors):
        dist = torusDistance(x0, y0, x1, y1, size)

        if dist < r:
            # A neighbor have been found, return True
            return True
    # No neighbor have been found, return false
    return False
 
def getAveragePosition(particles, r, x0, y0, i, size):

    xList = []
    yList = []

    for j,(x1,y1) in enumerate(particles):
        if j != i:
            dist = torusDistance(x0, y0, x1, y1, size)

            # Fetch x and y position of If particle within radius r 
            if dist <= r:
                xList.append(x1)
                yList.append(y1)
    # Find mean x,y of your neighbors
    if not xList or not yList:
        print(xList, yList)
    meanX = np.mean(xList)
    meanY = np.mean(yList)
    return meanX, meanY


def moveToMean(meanX, meanY, x2, y2, size):

    "Function that returns the closest direction from (meanX,meanY) to (x2,y2)"
    if abs(meanX - x2) < size - abs(meanX - x2):
        moveX = meanX - x2
    else:
        moveX = (meanX - x2) * -1

    if abs(meanY - y2) < size - abs(meanY - y2):
        moveY = meanY - y2
    else:
        moveY = (meanY - y2) * -1

    return moveX, moveY