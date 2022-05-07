#!/usr/bin/python

# File with functions needed for the geometry of the model. Inspired by: git@github.com:fskerman/vicsek_model.git

# Imports
from dis import dis
import numpy as np 
from math import atan2, pi, sin, cos, sqrt
import math

def euclideanDistance(x1, y1, x2, y2):
    ''' Function calculating the euclidean distance between coordinates (x,y) '''
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)

def vectorToAngle(v):
    ''' Functions that returns the angle of a given vector v '''
    x = v[0]
    y = v[1]
    return atan2(y,x)


def randomAngle():
    ''' Functions that generate random angle theta  between -pi - pi '''		
    theta = np.random.uniform(-pi,pi)
    return theta


def unitVector(v1, v2):
    ''' Function generating a unit vector '''
    vector = v1 - v2
    dist = euclideanDistance(v1[0], v1[1], v2[0],v2[1])
    uv = vector / dist
    return uv



def angleToVector(theta):
    ''' Function that returns the angle unit vector '''

    x = cos(theta)
    y = sin(theta)

    # Unit vector transformation
    v1 = np.array([x,y])
    v2 = np.array([0,0])
    uv = unitVector(v1,v2)

    return uv

def torusDistance(x1, y1, x2, y2, size):
    ''' Function returning the euclidean distance between (x,y) coordinates on a torus '''
    x_diff = min(abs(x1 - x2), size - abs(x1 - x2))
    y_diff = min(abs(y1 - y2), size - abs(y1 - y2))
    
    distance = sqrt(x_diff**2 + y_diff**2)
    angle = atan2(y_diff, x_diff) + math.pi
    return distance, angle

def predatorSense(x1, y1, x2, y2, size):
    ''' Function returning the euclidean distance between (x,y) coordinates on a torus '''

    # Difference in x and y axis both regular and torus
    x_diff = ((x1 - x2), size - (x1 - x2))
    y_diff = ((y1 - y2), size - (y1 - y2))

    # Calculate distance
    x_diffAbs = (abs(x1 - x2), size - abs(x1 - x2))
    y_diffAbs = (abs(y1 - y2), size - abs(y1 - y2))

    # Fetch Minimum distance
    minDistX = min(x_diffAbs)
    minDistY = min(y_diffAbs)

    # Fetch index of the minimum distance
    minIndexX= x_diffAbs.index(minDistX)
    minIndexY= y_diffAbs.index(minDistY)

    # Move in the direction of minimum distance
    moveX = x_diff[minIndexX]
    moveY = y_diff[minIndexY]
    
    return  moveX, moveY