
# Import and create graph of the football network provided by NetworkX
import urllib.request
import io
import zipfile
import numpy as np
import random

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

def initializeNetwork(network):
    randomPeople = random.sample(range(1, network.number_of_nodes()), 2)
    for i, node in enumerate(network.nodes.data()):
        if i == randomPeople[0]:
            node[1]["value"] = sharer
        elif i == randomPeople[1]:
            node[1]["value"] = bored
        else:
            node[1]["value"] = 0

    return network
    

def plotNetwork(network,title):
    colorMap = []
    for node in network.nodes.data():
        if node[1]["value"] == resting:
            colorMap.append('black')
        elif node[1]["value"] == sharer:
            colorMap.append('green')
        else:
            colorMap.append('red')
    my_pos = nx.spring_layout(network, seed = 100)
    plt.figure()  
    plt.title(title)

    nx.draw(network, node_color=colorMap, pos = my_pos)

    plt.show()

def updateNetwork(network, p, q, r):

    # # Create new state
    newState = network.copy()

    for i, node in enumerate(network.nodes.data()):

        # Fetch the state of the node
        person = node[1]["value"]

        # Resting rule
        if person == resting and random.random() <= p:
            list(newState.nodes.data())[i][1]["value"] = sharer
            # print(f"Person is sharer next state")
            continue
    
        # Sharer rule
        if person == sharer and random.random() <= q:
            # Fetch neighbors
            neighbors = list(all_neighbors(network, node[0]))
            
            # Pick random neighbor
            randomNeighbour = random.sample(neighbors,1)[0]

            # If that person is resting then they will now become a sharer
            if network.nodes.data()[randomNeighbour]["value"] == resting:
                newState.nodes.data()[randomNeighbour]["value"] = sharer
                continue

            # However, if the person they pick is bored, then the sharer will lose interest and become bored too
            elif network.nodes.data()[randomNeighbour]["value"] == bored:
                newState.nodes.data()[node[0]]["value"] = bored
                continue
        
        # Bored rule
        if person == bored and random.random() <= r:

             # Fetch neighbors
            neighbors = list(all_neighbors(network, node[0]))
            
            # Pick random neighbor
            randomNeighbour = random.sample(neighbors,1)[0]

            #If that person is resting then the bored person will now become resting
            if network.nodes.data()[randomNeighbour]["value"] == resting:
                newState.nodes.data()[node[0]]["value"] = resting
                continue

             # Otherwise they will continue to be bored
            else:
                newState.nodes.data()[node[0]]["value"] = bored
                continue





    return newState



if __name__ == "__main__":

    # G = nx.complete_graph(4)

    # Network 
    network = initializeNetwork(G)
    plotNetwork(network, "Initial state of network")




    # Set simulation parameters:
    p = 0.1
    q = 0.1
    r = 0.3
    numberOfSimulations = 200

    plt.ion()
    for i in range(numberOfSimulations):

        network = updateNetwork(network, p, q, r)
        plt.close()
        plotNetwork(network, f"Network at iteration {i+1}")
        plt.show()
        plt.pause(0.01)

