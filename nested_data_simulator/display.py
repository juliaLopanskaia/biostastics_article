import numpy as np
from math import *
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import matplotlib.ticker as ticker

def display_data(data_exp, data_control, N_clusters:int, N_per_cluster:int):
    """ display data (all experiments and means per clusters)

    INPUT: experimental data (matrix) & control data (matrix)
           the number of clusters and the number of experiments per clusters

    OUTPUT: None """
    #ipdb.set_trace()
    data_exp_mean = data_exp.mean(axis=0)
    data_control_mean = data_control.mean(axis=0)
    fig, ax = plt.subplots()
    fig = plt.plot(np.ones((N_per_cluster,1))+0.05/np.sqrt(N_clusters)*np.random.randn(N_per_cluster,1), data_exp,'.',markersize=6)

    colord = []
    for i in range(len(fig)):
        colord.append(fig[i].get_color())
    col=colord

    plt.scatter(np.ones(N_clusters), data_exp_mean, 1000, col,'+',lineWidths=3)

    arr_control=2*np.ones((N_per_cluster,1))+0.05/np.sqrt(N_clusters)*np.random.randn(N_per_cluster,1)
    for i in range(N_per_cluster):
        for j in range(N_clusters):
            plt.plot(arr_control[i], data_control[i][j],'.',markersize=6,color=col[j])
    plt.scatter(2*np.ones(N_clusters), data_control_mean, 1000, col,'+',lineWidths=3)

    ax.set_xlim(0,3)
    plt.show()



def display_heatmap(probability, MAX_N_clusters, MAX_N_per_cluster, scaleMax=1):
    """ INPUT: probability is a matrix
        scaleMax - the limitation of heatmap scale (optional, use if you want
        to compare several figures)

        OUTPUT: heatmap figure """
    CLUSTERS = np.array([i for i in range(2,MAX_N_clusters+1)])
    PER_CLUSTER = np.array([i for i in range(2,MAX_N_per_cluster+1)])
    ax = sns.heatmap(probability.T, xticklabels = CLUSTERS, yticklabels = PER_CLUSTER, vmin = 0, vmax = scaleMax)
    ax.invert_yaxis()
    plt.xlabel('number of clusters')
    plt.ylabel('number of measurements')
    #plt.title('All')
    plt.show()


def display_graph(probability, ICC, label):
    fig, ax = plt.subplots()
    for i in range(len(probability[:,1])):
        ax.scatter(ICC, probability[i,:], label = label[i])
    ax.legend()
    plt.xlabel('ICC')
    #  Устанавливаем интервал основных делений:
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    #  Устанавливаем интервал вспомогательных делений:
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    #  Тоже самое проделываем с делениями на оси "y":
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.patch.set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel('ICC')
    plt.ylabel('Probability of error')
    plt.show()
