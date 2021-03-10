import matplotlib.pyplot as plt
import numpy as np
from math import *

import ipdb



def display_data(data_exp, data_control, N_clusters:int, N_per_cluster:int):
    """ display data (all experiments and means per clusters)

    INPUT: experimental data (matrix) & control data (matrix)
           the number of clusters and the number of experiments per clusters

    OUTPUT: None """
    #ipdb.set_trace()
    data_exp_mean = data_exp.mean(axis=0)
    data_control_mean = data_control.mean(axis=0)
    fig, ax = plt.subplots()
    fig = plt.plot(np.ones((N_per_cluster,1))+0.05/sqrt(N_clusters)*np.random.randn(N_per_cluster,1), data_exp,'.',markersize=6)

    colord = []
    for i in range(len(fig)):
        colord.append(fig[i].get_color())
    col=colord

    plt.scatter(np.ones(N_clusters), data_exp_mean, 1000, col,'+',lineWidths=3)

    arr_control=2*np.ones((N_per_cluster,1))+0.05/sqrt(N_clusters)*np.random.randn(N_per_cluster,1)
    for i in range(N_per_cluster):
        for j in range(N_clusters):
            plt.plot(arr_control[i], data_control[i][j],'.',markersize=6,color=col[j])
    plt.scatter(2*np.ones(N_clusters), data_control_mean, 1000, col,'+',lineWidths=3)

    ax.set_xlim(0,3)
    plt.show()

data_exp = np.matrix([[1, 1.1], [1.2, 1.3]])
data_control = np.matrix([[1, 1.1], [1.2, 1.3]])
display_data(data_exp, data_control, 2, 2)