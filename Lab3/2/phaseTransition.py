
# Import and create graph of the football network provided by NetworkX
from cProfile import label
import urllib.request
import io
import zipfile
import numpy as np
import random
from Lattice import *
from Network import *

import matplotlib.pyplot as plt
import networkx as nx
from networkx import all_neighbors
from sklearn import neighbors

url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"

sock = urllib.request.urlopen(url)  # open URL
s = io.BytesIO(sock.read())  # read into BytesIO "file"
sock.close()

zf = zipfile.ZipFile(s)  # zipfile object
txt = zf.read("football.txt").decode()  # read info file
gml = zf.read("football.gml").decode()  # read gml data
# throw away bogus first line with # from mejn files
gml = gml.split("\n")[1:]
G = nx.parse_gml(gml)  # parse gml data

# print(txt)
# print degree for each team - number of games
# for n, d in G.degree():
#     # print(f"{n:20} {d:2}")

options = {"node_color": "black", "node_size": 50, "linewidths": 0, "width": 0.1}

pos = nx.spring_layout(G, seed=1969)  # Seed for reproducible layout
# nx.draw(G, pos, **options)
# plt.show()
# nx.info(G)


# State to number representation
resting = 0
sharer = 1
bored = 2

def numberOfBored():
    N = 10

    # Set simulation parameters:
    # p = 0.001
    q = 0.01
    r = 0.01

    pVector = np.linspace(0,0.1,20)
    numberOfSimulations = 50
    numberOfSteps = 500
    numberOfBoredListLattice = np.linspace(0, 1, numberOfSimulations*pVector.size)
    numberOfBoredListNetwork = np.linspace(0, 1, numberOfSimulations*pVector.size)

    for count, pSim in enumerate(pVector):
        
        p = pSim
        for sim in range(numberOfSimulations):

            # Initiliaze network and lattice
            network = initializeNetwork(G)
            population = initialiazePopulation(N)

            for j in range(numberOfSteps):

                # Update network and lattice to last time step
                network = updateNetwork(network, p, q, r)
                population = updatePopulation(population, p, q, r)

            
            # Simulate model
            numberOfBoredLattice = calculateNumberOfBoredLattice(population)/(N*N)
            numberOfBoredNetwork = calculateNumberOfBoredNetwork(network)/G.number_of_nodes()
            numberOfBoredListLattice[count*numberOfSimulations+sim] = numberOfBoredLattice
            numberOfBoredListNetwork[count*numberOfSimulations+sim] = numberOfBoredNetwork
            print(f"Percentage number of bored for lattice with p = {p} at sim {sim} is: {numberOfBoredLattice}")
            print(f"Percentage Number of bored for network with p = {p} at sim {sim} is: {numberOfBoredNetwork}")

    pVector = pVector.repeat(numberOfSimulations)
    plt.figure(3)
    plt.hist2d(pVector, numberOfBoredListLattice, bins=10)
    plt.xlabel("p")
    plt.ylabel(f"Number of bored (%)")
    plt.title("Phase transition plot of number of bored vs p")
    plt.colorbar()

    plt.figure(4)
    plt.hist2d(pVector, numberOfBoredListNetwork, bins=10)
    plt.xlabel("p")
    plt.ylabel(f"Number of bored (%)")
    plt.title("Phase transition plot of number of bored vs p")
    plt.colorbar()
    plt.show()

def numberOfResting():
    N = 10

    # Set simulation parameters:
    # p = 0.001
    q = 0.01
    r = 0.01
   
    pVector = np.linspace(0,0.1,10)
    numberOfSimulations = 10
    numberOfSteps = 500
    numberOfRestingListLattice = np.linspace(0, 1, numberOfSimulations*pVector.size)
    numberOfRestingListNetwork = np.linspace(0, 1, numberOfSimulations*pVector.size)

    for count, pSim in enumerate(pVector):
        
        p = pSim
        for sim in range(numberOfSimulations):

            # Initiliaze network and lattice
            network = initializeNetwork(G)
            population = initialiazePopulation(N)

            for j in range(numberOfSteps):

                # Update network and lattice to last time step
                network = updateNetwork(network, p, q, r)
                population = updatePopulation(population, p, q, r)

            
            # Simulate model
            numberOfRestingLattice = calculateNumberOfRestingLattice(population)/(N*N)
            numberOfRestingNetwork = calculateNumberOfRestingNetwork(network)/G.number_of_nodes()
            numberOfRestingListLattice[count*numberOfSimulations+sim] = numberOfRestingLattice
            numberOfRestingListNetwork[count*numberOfSimulations+sim] = numberOfRestingNetwork
            print(f"Percentage number of resting for lattice with = {p} at sim {sim} is: {numberOfRestingLattice}")
            print(f"Percentage Number of resting for network with = {p} at sim {sim} is: {numberOfRestingNetwork}")

    pVector = pVector.repeat(numberOfSimulations)
    plt.figure(3)
    plt.hist2d(pVector, numberOfRestingListLattice, bins=10)
    plt.xlabel("q")
    plt.ylabel(f"Number of resting (%)")
    plt.title("Phase transition plot of number of resting vs q")
    plt.colorbar()

    plt.figure(4)
    plt.hist2d(pVector, numberOfRestingListNetwork, bins=10)
    plt.xlabel("q")
    plt.ylabel(f"Number of resting (%)")
    plt.title("Phase transition plot of number of resting vs q")
    plt.colorbar()
    plt.show()


def numberOfSharers():
    N = 10

    # Set simulation parameters:
    p = 0.001
    # q = 0.01
    r = 0.01
   
    qVector = np.linspace(0,0.1,20)
    numberOfSimulations = 50
    numberOfSteps = 500
    numberOfSharersListLattice = np.linspace(0, 1, numberOfSimulations*qVector.size)
    numberOfSharersListNetwork = np.linspace(0, 1, numberOfSimulations*qVector.size)

    for count, qSim in enumerate(qVector):
        
        q = qSim
        for sim in range(numberOfSimulations):

            # Initiliaze network and lattice
            network = initializeNetwork(G)
            population = initialiazePopulation(N)

            for j in range(numberOfSteps):

                # Update network and lattice to last time step
                network = updateNetwork(network, p, q, r)
                population = updatePopulation(population, p, q, r)

            
            # Simulate model
            numberOfSharersLattice = calculateNumberOfSharersLattice(population)/(N*N)
            numberOfSharersNetwork = calculateNumberOfSharersNetwork(network)/G.number_of_nodes()
            numberOfSharersListLattice[count*numberOfSimulations+sim] = numberOfSharersLattice
            numberOfSharersListNetwork[count*numberOfSimulations+sim] = numberOfSharersNetwork
            print(f"Percentage number of sharers for lattice with = {q} is: {numberOfSharersLattice}")
            print(f"Percentage Number of sharers for network with = {q} is: {numberOfSharersNetwork}")

    qVector = qVector.repeat(numberOfSimulations)
    plt.figure(3)
    plt.hist2d(qVector, numberOfSharersListLattice, bins=10)
    plt.xlabel("q")
    plt.ylabel(f"Number of sharers (%)")
    plt.title("Phase transition plot of number of sharers (%) vs q")
    plt.colorbar()

    plt.figure(4)
    plt.hist2d(qVector, numberOfSharersListNetwork, bins=10)
    plt.xlabel("q")
    plt.ylabel(f"Number of sharers (%)")
    plt.title("Phase transition plot of number of sharers (%) vs q")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":

    # numberOfSharers()
    # numberOfResting()
    numberOfBored()
