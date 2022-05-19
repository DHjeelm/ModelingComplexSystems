from simulation import Simulation


def flockBehavior():

    simulation = Simulation.initEmptyPopulation()

    print(simulation.countEatenPrey())
    timeStep = 0.01
    T = 4


if __name__ == "__main__":
    flockBehavior()
