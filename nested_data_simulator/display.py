import numpy as np
from math import *
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

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



def display_heatmap(probability, MAX_N_clusters, MAX_N_per_cluster):
    """ INPUT: probability is a matrix
        OUTPUT: heatmap figure """
    CLUSTERS = np.array([i for i in range(1,MAX_N_clusters)])
    PER_CLUSTER = np.array([i for i in range(1,MAX_N_per_cluster)])
    ax = sns.heatmap(probability.T, xticklabels = CLUSTERS, yticklabels = PER_CLUSTER)
    ax.invert_yaxis()
    plt.xlabel('number of clusters')
    plt.ylabel('number of measurements')
    #plt.title('All')
    plt.show()


def display_graph(probability, ICC):
    fig, ax = plt.subplots()
    ax.scatter(ICC, probability, label='All')
    ax.legend()
    plt.xlabel('ICC')
    plt.show()
