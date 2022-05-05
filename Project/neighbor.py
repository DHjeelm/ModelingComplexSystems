#!/usr/bin/python
import numpy as np
from sklearn import neighbors
from geometry import *

# File with functions needed for neighbor calculation of the model. Inspired by: git@github.com:fskerman/vicsek_model.git

def getClosestNeighbor(particles, r, x0, y0, i):
    ''' Function returning the index of the closest neighbor'''

    # Remove yourself as a neighbor
    checkNeighbors = [x for l,x in enumerate(particles) if l!=i] 
    neighbors = []
    checkNeighbors = [x for l,x in enumerate(particles) if l!=i] 
    for j,(x1,y1) in enumerate(checkNeighbors):
        dist = torusDistance(x0, y0, x1, y1)
        if dist < r:
            neighbors.append(dist)
        else:
            neighbors.append(float('inf'))
        
    # Get minimum value and index
    minDistance = min(neighbors)

    minIndex = neighbors.index(minDistance)


    return minIndex, minDistance


def getNeighbors(particles, r, x0, y0):
    ''' Function returning a list of indices for all neighbors. It includes itself as a neighor so it will be included in average '''

    neighbors = []

    for j,(x1,y1) in enumerate(particles):
        dist = torusDistance(x0, y0, x1, y1)

        if dist < r:
            neighbors.append(j)

    return neighbors


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




