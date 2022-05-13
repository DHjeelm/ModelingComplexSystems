from painter_play import *
from runner import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mPatches
import random

def plotChromosome(score, xpos, ypos, room):
    plotRoom = room

    cmap = ListedColormap(["white", "green", "red"])
    notPainted = mPatches.Patch(color="white", label="Not painted")
    paintedOnce = mPatches.Patch(color="green", label="Painted once")
    paintedSeveral = mPatches.Patch(color="red", label="Painted several times")
    # plt.ion()
    for i in range(len(xpos)):
        # If painter have been on position once
        
        if plotRoom[xpos[i]-1,ypos[i]-1] == 0:
            plotRoom[xpos[i]-1,ypos[i]-1] = 1

        # Painter have been on this position more than once
        else:
            plotRoom[xpos[i]-1,ypos[i]-1] = 2
        if i == len(xpos)-1:
            # plt.close()
            plt.figure(1, figsize=(18, 8))
            plt.legend(handles=[paintedSeveral, paintedOnce,
                                notPainted], loc="best")
            plt.imshow(plotRoom, vmin=0, vmax=len(cmap.colors), cmap=cmap)
            plt.text(ypos[i]-1, xpos[i]-1, str("Painter"), va='center', ha='center', size=5)
            # plt.imshow(plotRoom)
            plt.yticks(color="w")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title(f"Chromosome with score {round(score,2)}")
            plt.show()
            # plt.pause(0.001)



room=np.zeros((20,40))

bestChromosome = [1, 1, 0, 3, 1, 1, 0, 2, 1, 1, 2, 2, 0, 0, 1, 3, 2, 2, 2, 2, 2, 1, 3, 1, 2, 2, 2, 0, 2, 2, 2, 3, 2, 3, 0, 2, 2, 2, 3, 3, 2, 1, 3, 2, 1, 0, 3, 3, 2, 2, 2, 1, 3, 1]
# Run painter
score, xpos, ypos = painter_play(bestChromosome, room)
# Plot painter
plt.plot([1,2,3],[1,2,3])
plt.show()
plotChromosome(score, xpos, ypos, room)