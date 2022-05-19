from setup import *
from typing import List
from vector import Vector2
from numpy import float64

import numpy as np
import math


class Boid:
    def __init__(self, location: Vector2, heading: Vector2, turnSpeed: float64, speed=1) -> None:
        self.idealHeading: Vector2 = None
        self.location = location
        self.heading = heading
        self.turnSpeed = turnSpeed
        self.speed = speed
        self.deathCount = 0

    def getBoidsInSight(self, population: List['Boid']) -> List['Boid']:
        neighbors = []
        # print(f"My location: {self.location}")
        for boid in population:
            if boid is not self:
                if self.getDistanceTo_torus(boid) < self.sightRadius:
                    neighbors.append(boid)

        return neighbors

    def applyNewState(self, timeStep: float64):
        if self.idealHeading:
            self.idealHeading.normalizeSelf()
            self.heading += (self.idealHeading - self.heading) * self.turnSpeed
            self.heading.normalizeSelf()

        self.heading.rotate(np.random.uniform(-math.pi, math.pi) * .2)
        self.location += self.heading * self.speed * timeStep

    def die(self):
        self.location = Vector2.initRandom()
        self.idealHeading = Vector2.initRandom_normalized()
        self.deathCount += 1

    def getDistanceTo_torus(self, boid: 'Boid') -> float64:
        return math.sqrt(min(abs(self.location.x - boid.location.x), size - abs(self.location.x - boid.location.x))**2 + min(abs(self.location.y - boid.location.y), size - abs(self.location.y - boid.location.y))**2)

    def getDirectionTo_torus(self, boid: 'Boid') -> Vector2:
        "Function that returns the closest direction from self to another boid on a torus shape"
        x = boid.location.x
        y = boid.location.y

        x1 = self.location.x
        y1 = self.location.y
        if abs(x1 - x) < size - abs(x1 - x):
            moveX = x - x1
        else:
            moveX = (x - x1) * -1

        if abs(y1 - y) < size - abs(y1 - y):
            moveY = y - y1
        else:
            moveY = (y - y1) * -1

        result = Vector2(moveX, moveY)
        result.normalizeSelf()
        return result

    def checkBoundaryCond(self):

        # Assure correct boundaries
        if self.location.x < 0:
            self.location.x = size + self.location.x

        if self.location.x > size:
            self.location.x = self.location.x - size

        if self.location.y < 0:
            self.location.y = size + self.location.y

        if self.location.y > size:
            self.location.y = self.location.y - size
