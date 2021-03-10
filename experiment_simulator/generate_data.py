import numpy as np

def generate_data(true_mean:float, inter_day_SD:float, intra_day_SD:float, N_clusters:int, N_per_cluster:int):
    """ This function generates data. It randomly calculates the value for
    experiments (the variation is set with SD). Data here has two levels of
    hierachy: experiments per cluster and the number of cluster.

    INPUT: true value of measurements, cluster-to-cluster variability,
    experiment-to-experiment variability (inside a cluster), the number of
    clusters, the number of experiments per cluster

    OUTPUT: data - matrix of data (0 axis is experimental values per cluster;
    1 axis is clusters) """
    # generate matrix with clusters and experiments per cluster
    data = true_mean + inter_day_SD*np.random.randn(1,N_clusters) + intra_day_SD*np.random.randn(N_per_cluster,N_clusters);
    # calculate mean values of clusters (per cluster means)
    #mean_cluster = np.mean(data,axis=0);
    return data
