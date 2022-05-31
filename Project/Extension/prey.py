from predator import Predator
from boid import Boid
from vector import Vector2
from typing import List
from numpy import float64
import math
import numpy as np


class Prey(Boid):
    def __init__(self, location: Vector2, heading: Vector2, sightRadius: float64, speed=1, turnSpeed=.3, useAlign=True, useCohesion=True, useSeparation=True) -> None:
        super().__init__(location, heading, turnSpeed, speed)
        self.sightRadius = sightRadius
        self.useAlign = useAlign
        self.useCohesion = useCohesion
        self.useSeparation = useSeparation
        self.avoidPredators = True

    def move(self, predators: List[Predator], preyPopulation: List['Prey'], timeStep):

        # Find closest predator
        closestPredator = None
        closestDistance = math.inf

        for boid in predators:

            distanceToPredator = self.getDistanceTo_torus(boid)
            if distanceToPredator < self.sightRadius and distanceToPredator < closestDistance:
                closestPredator = boid
                closestDistance = distanceToPredator

        if closestPredator:
            # print("Running away")
            direction = self.getDirectionTo_torus(closestPredator)
            self.idealHeading = direction * -1

        else:
            neighbors = self.getBoidsInSight(preyPopulation)

            if len(neighbors) > 0:
                avgLocation = Vector2(x=np.mean([boid.location.x for boid in neighbors]), y=np.mean(
                    [boid.location.y for boid in neighbors]))

                avgHeading = Vector2(
                    x=np.mean([v.heading.x for v in self.getBoidsInSight(population=neighbors)]), y=np.mean([v.heading.y for v in self.getBoidsInSight(population=neighbors)]))

                separation = Vector2(0, 0)
                if len(neighbors) > 1:
                    for boid in neighbors:
                        dir = self.getDirectionTo_torus(boid=boid)
                        separation += dir * \
                            (1/self.getDistanceTo_torus(boid=boid))

                    separation.normalizeSelf()

                # print("Aligning")

                self.idealHeading = Vector2(0, 0)
                if self.useCohesion:
                    self.idealHeading += (
                        avgLocation - self.location) * .4
                if self.useAlign:
                    self.idealHeading += avgHeading * .2
                if self.useSeparation:
                    self.idealHeading += separation * 0.01

                if self.idealHeading.length() == 0:
                    self.idealHeading = None
