#!/usr/bin/python
import numpy as np
from geometry import *

# File with functions needed for neighbor calculation of the model. Inspired by: git@github.com:fskerman/vicsek_model.git

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




