from dfr_simulation_template import dfr_simulation
import numpy as np
from matplotlib import pyplot as plt

video = str("silenceOfTheLambs_verbose")
# obtain data
DFR = np.zeros((4, 3))
for i in range(0, 4):
    if i == 0:
        loss_probability = 1e-4
    elif i == 1:
        loss_probability = 4e-4
    elif i == 2:
        loss_probability = 7e-4
    else:
        loss_probability = 1e-3
    for j in range(0, 3):
        if j == 0:
            fec = False
            ci = False
        elif j == 1:
            fec = True
            ci = False
        else:
            fec = True
            ci = True
        DFR[i][j] = dfr_simulation(random_seed=777, num_frames=10000, loss_probability=loss_probability,
                                   video_trace=video, fec=fec, ci=ci)

colors = ['red', 'blue', 'green', 'black']
labels = ['No FEC, NO ci', 'FEC, NO ci', 'FEC, ci']
[m, n] = np.shape(DFR)
x = [1e-4, 4e-4, 7e-4, 1e-3]
# plot figure
for i in range(0, n):
    plt.scatter(x[0], DFR[0][i], color=colors[0], marker='o')
    plt.scatter(x[1], DFR[1][i], color=colors[1], marker='^')
    plt.scatter(x[2], DFR[2][i], color=colors[2], marker='x')
    plt.scatter(x[3], DFR[3][i], color=colors[3], marker='h')
    #
    array = np.array(DFR)
    plt.plot(x, array[:, i], label=labels[i], linestyle='--')

plt.xlabel('loss probability')
plt.ylabel('DFR')
plt.title('DFR under loss probability:1e-4, 4e-4, 7e-4 or 1e-3')
plt.legend()
plt.show()
