
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

def averageNumberOfSharers():
    N = 10

    # Set simulation parameters:
    p = 0.001
    q = 0.5
    r = 0.01
   
    numberOfSimulations = 50
    numberOfSteps = 500

    numberOfSharersMatrixLattice = np.zeros((numberOfSimulations, numberOfSteps))
    numberOfSharersMatrixNetwork = np.zeros((numberOfSimulations, numberOfSteps))


    for i in range(numberOfSimulations):

        # Initiliaze network and lattice
        network = initializeNetwork(G)
        population = initialiazePopulation(N)

        for j in range(numberOfSteps):

            # Update network and lattice
            network = updateNetwork(network, p, q, r)
            population = updatePopulation(population, p, q, r)

            # if j == 300:
            #     plotNetwork(network, f"Test at simulation {i}")

            numberOfSharersMatrixLattice[i,j] = calculateNumberOfSharersLattice(population)
            numberOfSharersMatrixNetwork[i,j] = calculateNumberOfSharersNetwork(network)

    # Take average over the simulations
    numberOfSharersAverageListLattice =[]
    numberOfSharersAverageListNetwork =[]

    for k in range(numberOfSteps):
        numberOfSharersAverageListLattice.append(np.mean(numberOfSharersMatrixLattice[:,k])/(N*N))
        numberOfSharersAverageListNetwork.append(np.mean(numberOfSharersMatrixNetwork[:,k])/G.number_of_nodes())
        
    print(numberOfSharersAverageListNetwork)

    # Plotting
    plt.figure(1)
    plt.title(f"Average number of sharers as a function of time step in percent \n p = {p}, q = {q} and r = {r}")
    plt.plot(list(range(1, numberOfSteps+1)), numberOfSharersAverageListLattice, label = "Lattice")
    plt.plot(list(range(1, numberOfSteps+1)), numberOfSharersAverageListNetwork, label = "Network")
    plt.legend()
    plt.ylabel("Average number of sharers (%)")
    plt.xlabel("Time step")
    plt.show()

if __name__ == "__main__":

    averageNumberOfSharers()
