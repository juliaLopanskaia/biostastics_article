import numpy as np
from math import *
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import matplotlib.ticker as ticker
import pandas as pd




def display_data_Superplot(data_exp,data_control):
    """
    display data (all experiments and means per clusters) using Superplot
    INPUT: experimental data (matrix) & control data (matrix)
    OUTPUT: None
    """
    #Сreate dataframe from our data
    df = create_dataframe(data_exp,data_control)
    df_mean = create_dataframe(data_exp.mean(axis=1),data_control.mean(axis=1))

    #Plotting swarmplot for all points
    ax = sns.swarmplot(x='control or experiment', y='Arbitrary unit', hue="Ni", data=df, 				size=4,alpha=0.5, zorder=1, \
                  	palette=sns.color_palette("Spectral", \
                  	as_cmap=False,n_colors=len(df_mean)))

    #Plotting swarmplot for cluster averages
    ax = sns.swarmplot(x="control or experiment", y="Arbitrary unit", hue="Ni", size=10, 				edgecolor="k", linewidth=1, \
                       data=df_mean, alpha=1, zorder=2,\
                       palette=sns.color_palette("Spectral",\
                       as_cmap=False,n_colors=len(df_mean)))
    #Build the standard deviation from the mean values for the clusters
    ax.errorbar([0, 1], [data_exp.mean(axis=1).mean(), \
    			  data_control.mean(axis=1).mean()], \
    			  xerr=[0.2, 0.2],\
                	  color='black', elinewidth=2, \
                	  linewidth=0, zorder=3,  capsize=0)
    ax.errorbar([0, 1], [data_exp.mean(axis=1).mean(), data_control.mean(axis=1).mean()],  \
                yerr= [data_exp.mean(axis=1).std()/(np.sqrt(len(data_exp))),\
                	data_control.mean(axis=1).std()/(np.sqrt(len(data_control)))], \
                	color='black', elinewidth=2, \
                	linewidth=0, zorder=3,  capsize=4)
    ax.get_legend().remove()#Removing the legend
    plt.xlim(-0.7,1.7)#Setting the limits on the abscissa axis
    plt.ylim(0,2)#Setting the limits on the ordinate axis
    plt.xlabel('')#X-axis signature
    ax.patch.set_visible(False) #Invisible background
    ax.spines['right'].set_visible(False) #Invisible right line drawing boxing
    ax.spines['top'].set_visible(False)#Invisible top line drawing boxing
    plt.show()


def display_heatmap(probability, MAX_N_clusters, MAX_N_per_cluster, scaleMax=1):
    """ INPUT: probability is a matrix
        scaleMax - the limitation of heatmap scale (optional, use if you want
        to compare several figures)

        OUTPUT: heatmap figure """
    CLUSTERS = np.array([i for i in range(2,MAX_N_clusters+1)])
    PER_CLUSTER = np.array([i for i in range(2,MAX_N_per_cluster+1)])
    ax = sns.heatmap(probability.T, xticklabels = CLUSTERS, \
    		     yticklabels = PER_CLUSTER, vmin = 0, vmax = scaleMax)
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
    
    
    
    
    
def create_dataframe(data_exp,data_control):
    """
    Create dataframe from experimental data (matrix) & control data (matrix)
    INPUT: experimental data (matrix) & control data (matrix)
    OUTPUT: dataframe based on experimental data and control data
            dataframe['control or experiment'] - value from control data or experiment data
            dataframe['Arbitrary unit'] - this value
            dataframe['Ni'] - cluster number using end-to-end cluster numbering from experimental and control data
    """
    #Checking whether it is represented by a two-dimensional matrix, or otherwise - a one-dimensional matrix.
    #Depending on this, we set the size of the data. This is for an experimental data
    if type(data_exp[0]) == np.ndarray:
        N_clusters_exp = len(data_exp)
        N_per_cluster_exp = len(data_exp[0])
        Ni_exp = np.array([(i // N_per_cluster_exp) for i in range(len(data_exp.reshape(-1)))])
    else:
        N_clusters_exp = 1
        N_per_cluster_exp = len(data_exp)
        Ni_exp = np.array([i for i in range(len(data_exp.reshape(-1)))])
    #Likewise for control data
    if type(data_control[0]) == np.ndarray:
        N_clusters_control = len(data_control)
        N_per_cluster_control = len(data_control[0])
        Ni_control = np.array([(i // N_per_cluster_control)+1 +max(Ni_exp) for i in range(len(data_control.reshape(-1)))])
    else:
        N_clusters_control = 1
        N_per_cluster_control = len(data_control)
        Ni_control = np.array([(i)+1+max(Ni_exp)  for i in range(len(data_control.reshape(-1)))])
    #Create a list with data belonging to a specific group
    x_exp = np.array(['experiment' for i in range(len(data_exp.reshape(-1)))])
    x_control = np.array(['control' for i in range(len(data_control.reshape(-1)))])
    #create a dictionary corresponding to the future dataframe
    d = {'control or experiment': np.concatenate((x_control,x_exp), axis=0),
         'Arbitrary unit' : np.concatenate((data_exp.reshape(-1),data_control.reshape(-1)), axis=0),
         'Ni': np.concatenate((Ni_exp,Ni_control), axis=0)}
    #Create dataframe
    df = pd.DataFrame(data=d)
    return df

