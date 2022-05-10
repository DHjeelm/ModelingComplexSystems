# Code is modified from https://github.com/alsignoriello/vicsek_model

import os, sys
from turtle import distance
from cv2 import mean

import numpy as np
import matplotlib.pyplot as plt

from math import pi, sqrt, cos, sin, atan2

#for videos
import moviepy.video.io.ImageSequenceClip




#---------- Geometric functions -----------------------------

def vector_2_angle(v):
    x = v[0]
    y = v[1]
    return atan2(y,x)


# generate random angle theta between -pi - pi
def rand_angle():       
    theta = np.random.uniform(-pi,pi)
    return theta


# returns angle unit vector
def angle_2_vector(theta):
    x = cos(theta)
    y = sin(theta)
    
    # transform to unit vector
    v1 = np.array([x,y])
    #v2 = np.array([0,0])
    #uv = unit_vector(v1,v2)

    uv = v1/ euclidean_distance(v1[0], v1[1], 0, 0)

    return uv


# Euclidean distance between (x,y) coordinates
def euclidean_distance(x1, y1, x2, y2):
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)


# Euclidean distance between (x,y) coordinates on 1 x 1 torus
def torus_distance(x1, y1, x2, y2):
    x_diff = min(abs(x1 - x2), 1 - abs(x1 - x2))
    y_diff = min(abs(y1 - y2), 1 - abs(y1 - y2))
    return sqrt(x_diff**2 + y_diff**2)


def unit_vector(v1, v2):
    vector = v1 - v2
    dist = euclidean_distance(v1[0], v1[1], v2[0],v2[1])
    v1v2 = vector / dist
    return v1v2


#---------------------------------------------------------------------------------------

#-------------------------Functions of Neighbours --------------------------------------


# returns a list of indices for all neighbors
# includes itself as a neighor so it will be included in average
def get_neighbors(particles, r, x0, y0):

    neighbors = []

    for j,(x1,y1) in enumerate(particles):
        dist = torus_distance(x0, y0, x1, y1)

        if dist < r:
            neighbors.append(j)

    return neighbors


# average unit vectors for all angles
# return average angle by converting to vectors, using vector addition top-to-tail, then taking arc tan to get angle of resulting vector.
def get_average(thetas, neighbors):
    
    n_neighbors = len(neighbors)
    avg_vector = np.zeros(2)

    for index in neighbors:
        theta = thetas[index,0]
        theta_vec = angle_2_vector(theta)
        avg_vector += theta_vec

    avg_angle = vector_2_angle(avg_vector)

    return avg_angle

def get_average_position(particles, r, x0, y0, i):

    xList = []
    yList = []

    # Remove yourself from the list
    checkNeighbors = [x for l,x in enumerate(particles) if l!=i]

    for j,(x1,y1) in enumerate(checkNeighbors):
        dist = torus_distance(x0, y0, x1, y1)

        # Fetch x and y position of If particle within radius r 
        if dist < r:
            xList.append(x1)
            yList.append(y1)
    # Find mean x,y of your neighbors
    meanX = np.mean(xList)
    meanY = np.mean(yList)
            
    return meanX, meanY

def check_if_any_neighbors(particles, r, x0, y0, i):
    # Remove yourself from the list
    checkNeighbors = [x for l,x in enumerate(particles) if l!=i]

    # Loop through particles and check if you have any neighbors
    for j,(x1,y1) in enumerate(checkNeighbors):
        dist = torus_distance(x0, y0, x1, y1)

        if dist < r:
            # A neighbor have been found, return True
            return True
    # No neighbor have been found, return false
    return False

def get_closest_particle(particles, r, x0, y0, i):
    ''' Function returning the distance closest particle'''

    # Remove yourself as a neighbor
    checkNeighbors = [x for l,x in enumerate(particles) if l!=i] 
    distances = []
    for j,(x1,y1) in enumerate(checkNeighbors):

        distances.append(torus_distance(x0, y0, x1, y1))

    # Get minimum distance
    minDistance = min(distances)

    # Return minimum value and index
    return minDistance

def move_to_average(meanX, meanY, x2, y2):
    '''Function calculating how to move (x,y) to move closer to average the average of your neighbors'''

    # Difference in x and y axis both regular and torus
    x_diff = ((meanX - x2), 1 - (meanX - x2))
    y_diff = ((meanY - y2), 1 - (meanY - y2))

    # Calculate distance
    x_diffAbs = (abs(meanX - x2), 1 - abs(meanX - x2))
    y_diffAbs = (abs(meanY - y2), 1 - abs(meanY - y2))

    # Fetch Minimum distance
    minDistX = min(x_diffAbs)
    minDistY = min(y_diffAbs)

    # Fetch index of the minimum distance
    minIndexX= x_diffAbs.index(minDistX)
    minIndexY= y_diffAbs.index(minDistY)

    # Move in the direction of minimum distance
    moveX = x_diff[minIndexX]
    moveY = y_diff[minIndexY]
    
    return  moveX, moveY


#-------------------------------------------------------------------------


def plot_vectors(coords, thetas):

	# generate random color for every particle
	colors = ["b", "g", "y", "m", "c", "pink", "purple", "seagreen",
			"salmon", "orange", "paleturquoise", "midnightblue",
			"crimson", "lavender"]

	
	for i, (x, y) in enumerate(coords):

		c = colors[i % len(colors)]

		# plot point
		plt.scatter(x, y, color = c, marker = ".")

		# plot tail
		theta = thetas[i]
		v = angle_2_vector(theta)
		x1 = x - (0.025 * v[0])
		y1 = y - (0.025 * v[1])
		plt.plot([x, x1], [y, y1], color=c)

	return



def save_plot(path, fname, eta):

    # axes between 0 and 1
    plt.axis([0, 1, 0, 1])

    # remove tick marks
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])

    # title 
    plt.title("η = %.2f" % eta)

    # save plot
    plt.savefig(os.path.join(path, fname[:-4]+".jpg"))
    plt.close()

    # clear for next plot
    plt.cla()

    return


# ------------------------------- RUNS FROM HERE -------------------------------

if __name__ == '__main__':

    # Eta
    # etas = np.linspace(0,1,50)
    # numberOfSimulations = 50
    # averageNearestNeighborList = np.linspace(0, 1, numberOfSimulations*etas.size)

    # r
    # rs = np.linspace(0,np.sqrt(0.5),50)
    # numberOfSimulations = 50
    # averageNearestNeighborList = np.linspace(0, 1, numberOfSimulations*rs.size)

    # T
    time = np.linspace(0.2,1,20)
    numberOfSimulations = 50
    averageNearestNeighborList = np.linspace(0, 1, numberOfSimulations*time.size)

    for count, tSim in enumerate(time):
        # etaSim = eta
        T = round(tSim,2)
        for sim in range(numberOfSimulations):

            
            N = 40         # num of particles
            eta = 0.2    # noise in [0,1], add noise uniform in [-eta*pi, eta*pi]
            r = 0.1       # radius
            delta_t = 0.01   # time step

            # Maximum time
            t = 0.0
            # T = 0.5 #was 2.0

            # Generate random particle coordinates
            # particles[i,0] = x
            # particles[i,1] = y
            particles = np.random.uniform(0, 1, size=(N, 2))

            # initialize random angles
            thetas = np.zeros((N, 1))
            for i, theta in enumerate(thetas):
                thetas[i, 0] = rand_angle()

            # Currently run until time ends
            while t < T:
                nearestNeighborListInner = []
                for i, (x, y) in enumerate(particles):

                    # If we have any neighbors, move to mean position of the neighboors
                    if (check_if_any_neighbors(particles, r, x, y, i)):

                        meanX, meanY = get_average_position(particles, r, x, y, i)

                        moveX, moveY = move_to_average(meanX, meanY, x, y)

                        # Update the theta
                        angle = vector_2_angle((moveX,moveY))

                        # Fetch random angle
                        n_angle = rand_angle()

                        # Multiply with eta
                        noise = eta * n_angle

                        thetas[i] = angle + noise

                        # Update position
                        particles[i,:] += delta_t * angle_2_vector(thetas[i])

                    # Move randomly
                    else:
                        # Fetch random angle
                        n_angle = rand_angle()

                        # Multiply with eta
                        noise = eta * n_angle

                        thetas[i] += noise

                        # Update position
                        particles[i,:] += delta_t * angle_2_vector(thetas[i])


                    # assure correct boundaries (xmax, ymax) = (1,1)
                    if particles[i, 0] < 0:
                        particles[i, 0] = 1 + particles[i, 0]

                    if particles[i, 0] > 1:
                        particles[i, 0] = particles[i, 0] - 1

                    if particles[i, 1] < 0:
                        particles[i, 1] = 1 + particles[i, 1]

                    if particles[i, 1] > 1:
                        particles[i, 1] = particles[i, 1] - 1
                        
                    # Calculate at last time step
                    if round(t,2) == round(T-delta_t,2):
                        # Caluclate nearest neighbor for particle        
                        nearestNeighborListInner.append(get_closest_particle(particles, r, x, y, i))
    
                # new time step
                t += delta_t
            # Average of the nearest neighbor for all the particles at last time step
            averageNearestNeighbor = np.mean(nearestNeighborListInner)
            # print(f"Average distance to nearest neighbor at simulation {sim} with eta {etaSim} is: {averageNearestNeighbor}")
            # print(f"Average distance to nearest neighbor at simulation {sim} with r {r} is: {averageNearestNeighbor}")
            print(f"Average distance to nearest neighbor at simulation {sim} with T {T} is: {averageNearestNeighbor}")
            averageNearestNeighborList[count*numberOfSimulations+sim] = np.mean(averageNearestNeighbor)

    # etas = etas.repeat(numberOfSimulations)
    # plt.figure(2)
    # plt.hist2d(etas, averageNearestNeighborList, bins=10)
    # plt.xlabel("\u03B7")
    # plt.ylabel(f"Average distance to nearest neighbor")
    # plt.title("Phase transition plot of average distance to nearest neighbor vs η")
    # plt.colorbar()

    # rs = rs.repeat(numberOfSimulations)
    # plt.figure(2)
    # plt.hist2d(rs, averageNearestNeighborList, bins=10)
    # plt.xlabel("r")
    # plt.ylabel(f"Average distance to nearest neighbor")
    # plt.title("Phase transition plot of average distance to nearest neighbor vs r")
    # plt.colorbar()

    rs = time.repeat(numberOfSimulations)
    plt.figure(2)
    plt.hist2d(rs, averageNearestNeighborList, bins=10)
    plt.xlabel("T")
    plt.ylabel(f"Average distance to nearest neighbor")
    plt.title("Phase transition plot of average distance to nearest neighbor vs T")
    plt.colorbar()


plt.show()