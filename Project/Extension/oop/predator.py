
from sklearn import neighbors
from boid import Boid
from vector import Vector2
from numpy import float64
import numpy as np
from typing import List
import math


class Predator(Boid):

    def __init__(self, location: Vector2, heading: Vector2, sightRadius: float64, speed=1.5, turnSpeed=.5, eatRadius=.03) -> None:
        super().__init__(location, heading, turnSpeed, speed)
        self.eatenPrey = 0
        self.sightRadius = sightRadius
        self.eatRadius = eatRadius
        self.avoidPredators = True


    def move(self, preyPopulation: List[Boid], predators: List['Predator'], timeStep: float64):
        """Find closest pray and target it. If no pray is found within search radius, avoid predators or randomly move"""

        closestPrey = None
        closestDistance = math.inf

        # Find closest pray
        for boid in preyPopulation:
            if type(boid) is not Predator:
                distanceToPrey = self.getDistanceTo_torus(boid)
                if distanceToPrey < self.sightRadius and distanceToPrey < closestDistance:
                    # print("I see you")
                    closestPrey = boid
                    closestDistance = distanceToPrey

        if closestPrey:
            if closestDistance < self.eatRadius:
                closestPrey.die()
                self.eatenPrey += 1
                # print("Pred ate prey")
                return

            # print("Hunting")

            # print(f"My location: {self.location}")
            # print(f"Target location: {closestPrey.location}")
            direction = self.getDirectionTo_torus(closestPrey)
            # print(f"Heading towards target: {direction}")
            self.idealHeading = direction

        elif self.avoidPredators:
            neighbors = self.getBoidsInSight(population=predators)
            if len(neighbors) > 0:
                separation = Vector2(0, 0)

                for boid in neighbors:
                    dir = self.getDirectionTo_torus(boid=boid)
                    separation += dir * \
                        (1/self.getDistanceTo_torus(boid=boid))

                # print("Aligning")

                self.idealHeading = separation * -1
