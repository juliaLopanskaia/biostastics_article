import matplotlib.pyplot as plt
import numpy as np
from math import *

def figure_generate(N_per_cluster, N_clusters, data_exp, mean_exp, data_control, mean_control):
    fig, ax = plt.subplots()
    fig = plt.plot(np.ones((N_per_cluster,1))+0.05/sqrt(N_clusters)*np.random.randn(N_per_cluster,1), data_exp,'.',markersize=6)

    colord = []
    for i in range(len(fig)):
        colord.append(fig[i].get_color())
    col=colord

    plt.scatter(np.ones(N_clusters), mean_exp, 1000, col,'+',lineWidths=3)

    arr_control=2*np.ones((N_per_cluster,1))+0.05/sqrt(N_clusters)*np.random.randn(N_per_cluster,1)
    for i in range(N_per_cluster):
        for j in range(N_clusters):
            plt.plot(arr_control[i], data_control[i][j],'.',markersize=6,color=col[j])
    plt.scatter(2*np.ones(N_clusters), mean_control, 1000, col,'+',lineWidths=3)

    ax.set_xlim(0,3)
    plt.show()
