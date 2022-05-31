

import math
from numpy import float64
from matplotlib import pyplot as plt
from typing import List
from prey import Prey, Predator
from simSetup import size
from vector import Vector2


class Simulation:

    def __init__(self, nPrey=40, nPred=20, speedPred=1.5, speedPrey=1, size=1, rSightPred=.2, rSightPrey=.3):

        self.nPrey = nPrey
        self.nPred = nPred
        self.speedPred = speedPred
        self.speedPrey = speedPrey
        self.preyPopulation: List[Prey] = []
        self.predators: List[Predator] = []

        for _ in range(self.nPrey):
            a = Prey(heading=Vector2.initRandom_normalized(),
                     location=Vector2.initRandom(), sightRadius=rSightPrey, speed=speedPrey)
            self.preyPopulation.append(a)

        for _ in range(nPred):
            self.predators.append(
                Predator(heading=Vector2.initRandom_normalized(), location=Vector2.initRandom(), sightRadius=rSightPred, speed=speedPred))

    @classmethod
    def initEmptyPopulation(cls) -> 'Simulation':
        pop = cls(nPred=0, nPrey=0)
        pop.nPrey = 0
        pop.nPred = 0
        pop.speedPred = 1.5
        pop.speedPrey = 1
        pop.preyPopulation = []
        pop.predators = []
        return pop

    def analysisHelper(self, values: List[int]):

        if 0 in values:
            self.setPreyAvoidPred(True)
        else:
            self.setPreyAvoidPred(False)
        if 1 in values:
            self.setPreyAlign(True)
        else:
            self.setPreyAlign(False)

        if 2 in values:
            self.setPreyCohesion(True)
        else:
            self.setPreyCohesion(False)

        if 3 in values:
            self.setPreySeparation(True)
        else:
            self.setPreySeparation(False)

    def setPreyAlign(self, value: bool):
        for prey in self.preyPopulation:
            prey.useAlign = value

    def setPreyCohesion(self, value: bool):
        for prey in self.preyPopulation:
            prey.useCohesion = value

    def setPreySeparation(self, value: bool):
        for prey in self.preyPopulation:
            prey.useSeparation = value

    def setPreyAvoidPred(self, value: bool):
        for prey in self.preyPopulation:
            prey.avoidPredators = value

    def initDebug(self):
        self.nPrey = 1
        self.nPred = 1
        self.rPrey = .3
        self.rPred = .2
        self.speedPred = 2
        self.speedPrey = 1
        self.preyPopulation: List[Prey] = [Prey(heading=Vector2.initRandom_normalized(),
                                                location=Vector2.initRandom(), sightRadius=self.rPrey, speed=self.speedPrey)]
        self.predators = [Predator(
            heading=Vector2(-1, 0), location=Vector2(.7, .5), sightRadius=self.rPred, speed=self.speedPred)]

    def simulate(self, timeStep: float64, duration: float64, showProgress=False, awaitWindowClose=True):

        if not awaitWindowClose:
            plt.ion()
        for i in range(math.floor(duration / timeStep)):

            for boid in self.predators:
                boid.move(self.preyPopulation, self.predators, timeStep)

            for boid in self.preyPopulation:
                boid.move(self.predators, self.preyPopulation, timeStep)

            for boid in self.predators + self.preyPopulation:
                boid.applyNewState(timeStep)
                boid.checkBoundaryCond()

            if (showProgress):
                plt.close()
                self.showState()
                plt.show()
                plt.pause(0.0001)

    def countEatenPrey(self):

        count = 0
        for boid in self.preyPopulation + self.predators:
            if type(boid) == Predator:
                count += boid.eatenPrey

        return count

    def showState(self):
        '''Function creating a plot for the current state of the model'''

        showSightRadius = False
        figure, axes = plt.subplots()

        for boid in self.predators + self.preyPopulation:

            # Determine if predator or not
            if type(boid) == Predator:
                c = "r"
            else:
                c = "g"

            # Plot a particle
            axes.scatter(boid.location.x, boid.location.y,
                         color=c, marker=".")

            if showSightRadius:
                circle = plt.Circle((boid.location.x, boid.location.y),
                                    boid.sightRadius - 0.01, color="b", fill=False)
                axes.add_artist(circle)

            # Plot the tail of the particle

            heading = boid.heading
            heading.normalizeSelf()

            tail = boid.location - (heading * 0.025)

            axes.axis([0, size, 0, size])

            x1 = boid.location.x - (0.025 * heading.x)
            y1 = boid.location.y - (0.025 * heading.y)
            axes.plot([boid.location.x, x1], [
                boid.location.y, y1], color=c)

        # textstr = f"Parameters:\n\nPrey:\nNumber of prey: {numberOfPrey}\netaPrey: {etaPrey}\nrPrey: {rPrey}\n\nPredator\nNumber of predators: {numberOfPredators}\netaPredator: {etaPredator}\nrPredator: {rPredator}\nrEat: {rEat}"
        # plt.text(0.02, 0.5, textstr, fontsize=10, transform=plt.gcf().transFigure)
        # plt.subplots_adjust(left=0.3)


if __name__ == "__main__":

    from vector import Vector2
    from prey import Prey, Boid
    from predator import Predator

    simulation = Simulation(nPred=3, nPrey=50)

    # simulation.initDebug()
    simulation.setPreyAvoidPred(False)
    simulation.simulate(timeStep=.01, duration=4,
                        showProgress=True, awaitWindowClose=False)

    print(simulation.countEatenPrey())
