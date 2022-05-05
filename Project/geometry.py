#!/usr/bin/python

# File with functions needed for the geometry of the model. Inspired by: git@github.com:fskerman/vicsek_model.git

# Imports
import numpy as np 
from math import atan2, pi, sin, cos, sqrt

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

# Euclidean distance between (x,y) coordinates on 1 x 1 torus
def torusDistance(x1, y1, x2, y2):
    ''' Function returning the euclidean distance between (x,y) coordinates on 1 x 1 torus '''
    x_diff = min(abs(x1 - x2), 1 - abs(x1 - x2))
    y_diff = min(abs(y1 - y2), 1 - abs(y1 - y2))
    return sqrt(x_diff**2 + y_diff**2)