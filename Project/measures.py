# File with all the measure for the model.
import math
import numpy as np
from neighbor import *

def calculatePolarisation(thetas, numberOfParticles):
    # Calculate cosine and sine sum
    cosSum = (np.sum(np.cos(thetas)))**2
    sinSum = (np.sum(np.sin(thetas)))**2
    # Polarisation
    polarisation = 1/numberOfParticles*np.sqrt(cosSum+ sinSum)

    return polarisation

def cohesion(particles, r, x0, y0, i, size):
    "Function  calculating the cohesion for an particle"
    # Remove yourself as a neighbor
    checkNeighbors = [x for l,x in enumerate(particles) if l!=i] 
    neighbors = []
    for j,(x1,y1) in enumerate(checkNeighbors):

        dist = torusDistance(x0, y0, x1, y1, size)

        if dist < r:
            neighbors.append(dist)
        
    # Calculate average distance to neighbors (cohesion)
    cohesion = np.mean(neighbors)
    return cohesion
