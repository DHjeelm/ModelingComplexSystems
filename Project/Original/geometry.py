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
    return atan2(y, x)


def randomAngle():
    ''' Functions that generate random angle theta  between -pi - pi '''
    theta = np.random.uniform(-pi, pi)
    return theta


def unitVector(v1, v2):
    ''' Function generating a unit vector '''
    vector = v1 - v2
    dist = euclideanDistance(v1[0], v1[1], v2[0], v2[1])
    uv = vector / dist
    return uv


def angleToVector(theta):
    ''' Function that returns the angle unit vector '''

    x = cos(theta)
    y = sin(theta)

    # Unit vector transformation
    v1 = np.array([x, y])
    v2 = np.array([0, 0])
    uv = unitVector(v1, v2)

    return uv


def torusDistance(x1, y1, x2, y2, size):
    ''' Function returning the euclidean distance between (x,y) coordinates on a torus '''
    x_diff = min(abs(x1 - x2), size - abs(x1 - x2))
    y_diff = min(abs(y1 - y2), size - abs(y1 - y2))
    distance = sqrt(x_diff**2 + y_diff**2)

    return distance


def predatorSense(x1, y1, x2, y2, size):

    "Function that returns the closest direction from (x1,y1) to (x2,y2)"
    if abs(x1 - x2) < size - abs(x1 - x2):
        moveX = x1 - x2
    else:
        moveX = (x1 - x2) * -1

    if abs(y1 - y2) < size - abs(y1 - y2):
        moveY = y1 - y2
    else:
        moveY = (y1 - y2) * -1

    return moveX, moveY
