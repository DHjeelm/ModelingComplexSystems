


from painter_play_walls import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mPatches
import random

def plotChromosome(score, xpos, ypos, room):
    plotRoom = room
    # print(room)

    cmap = ListedColormap(["white", "blue", "green", "red"])
    notPainted = mPatches.Patch(color="white", label="Not painted")
    paintedOnce = mPatches.Patch(color="green", label="Painted")
    paintedSeveral = mPatches.Patch(color="red", label="Painted several times")
    wall = mPatches.Patch(color="blue", label="Wall")

    # plt.ion()
    for i in range(len(xpos)):
        # If painter have been on position once
        if plotRoom[xpos[i],ypos[i]] == 0:
            plotRoom[xpos[i],ypos[i]] = 2

        # Painter have been on this position more than once
        else:
            plotRoom[xpos[i],ypos[i]] = 3
        if i == len(xpos)-1:
        # plt.close()
            plt.figure(1, figsize=(18, 8))
            plt.legend(handles=[paintedOnce, wall, notPainted, paintedSeveral], loc="best")
            plt.imshow(plotRoom, vmin=0, vmax=len(cmap.colors), cmap=cmap)
            plt.text(ypos[i], xpos[i], str("Painter"), va='center', ha='center', size=5)
            # plt.imshow(plotRoom)
            plt.yticks(color="w")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"Chromosome with score {round(score,2)}")
            plt.show()
        # plt.pause(0.0001)


if __name__ == "__main__":

    # Best chromosome on furniture
    bestChromosome = [0,3, 1,2, 1, 2, 2, 2, 0, 2, 1, 2, 2, 3, 1, 2, 1, 3, 2, 0, 1, 3, 2, 1, 2, 1, 1, 2, 2, 2, 3, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 3, 1, 3, 2, 0, 2, 2, 1, 0, 2, 0]
    trained = [0, 2, 3, 1, 0, 3, 3, 0, 0, 1, 1, 0, 0, 1, 2, 1, 2, 2, 2, 1, 3, 3, 3, 1, 1, 3, 2, 3, 0, 2, 3, 2, 2, 1, 2, 3, 1, 0, 1, 0, 2, 1, 1, 2, 1, 2, 3, 3, 2, 2, 0, 0, 1, 0]

    room=np.zeros((20,40))
    indices = [(m, n) for m in range(20) for n in range(40)]
    randomIndices = random.sample(indices, 100)
    for idx in randomIndices:
        room[idx[0], idx[1]] = 1

    score, xpos, ypos, env, plotEnv = painter_play_walls(trained, room, randomIndices)
   
    plotChromosome(score, xpos, ypos, plotEnv)

