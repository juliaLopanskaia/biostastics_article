import numpy as np
from scipy.stats import ttest_ind as ttest
from scipy.stats import t as tpdf
from math import *



def generate_data(true_value:float, inter_day_SD:float, intra_day_SD:float, \
                  N_clusters:int, N_per_cluster:int):
    """ This function generates data. It randomly calculates the value for
    experiments (the variation is set with SD). Data here has two levels of
    hierachy: experiments per cluster and the number of cluster.

    INPUT: true value of measurements, cluster-to-cluster variability,
    experiment-to-experiment variability (inside a cluster), the number of
    clusters, the number of experiments per cluster

    OUTPUT: data - matrix of data (0 axis is experimental values per cluster;
    1 axis is clusters) """
    # generate matrix with clusters and experiments per cluster
    data = true_value + inter_day_SD*np.random.randn(1,N_clusters) + \
           intra_day_SD*np.random.randn(N_per_cluster,N_clusters)
    return data



def adj_ttest(N_per_cluster:int, N_clusters:int, inter_day_SD:float, \
              intra_day_SD:float, data_exp_pooled:list, \
              data_control_pooled:list):
    N = N_per_cluster*N_clusters # total number of experiments
    ICC = inter_day_SD**2/(inter_day_SD**2 + intra_day_SD**2); # intraclass correlation calculation
    c = sqrt(((N - 2) - 2*(N_per_cluster-1)*ICC)/((N-2)*(1+(N_per_cluster-1)*ICC))) # correction factor for t-distribution     FIXME rename as items
    df = ((N-2)-2*(N_per_cluster-1)*ICC)**2/((N-2)*(1-ICC)**2+N_per_cluster-1*(N-2*N_per_cluster-1)*ICC+2*(N-2*N_per_cluster)*ICC*(1-ICC)) # corrected degrees of freedom
    s = sqrt(((N-1)*np.std(data_exp_pooled)**2+(N-1)*np.std(data_control_pooled)**2)/(2*N-2)) # standard deviation of two datasets
    t = abs(np.mean(data_exp_pooled) - np.mean(data_control_pooled))/(s*sqrt(1/N + 1/N)) # t-test
    ta = c*t # corrected t-test
    p_value = 2*sum(tpdf.pdf(np.arange(ta,100,0.001),df)*0.001) # p-value = integral of t-distribution probability function
    #print('P-value based on t-distribution probability function is {:2.2f}'.format(p_value))
    h = 1 # FIXME
    return h, p_value



def process_data(data_exp, data_control, N_per_cluster, N_clusters, \
                 inter_day_SD, intra_day_SD, data_method, ttest_method):
    """ This is the function to process data
    There are several types of processing
    By default it is use simple t-test on pooled data (ignore clustering)
    INPUT: 1) the parameters for data generating
            2) data_method = {‘pool’, ‘cluster’}, optional
               choose the type of data to process furter
               ( if 'pool', use the pooled data
               elif 'cluster_means' use the means of clusters )
            3) ttest_method = {'simple', 'adjusted'}, optional
               choose what type of ttest to apply For more information read methods.md
     """

    if data_method == 'pool': # use pooled data for processing
        # pool the data into a list:
        data_exp_pooled = data_exp.reshape(-1).tolist()
        data_control_pooled = data_control.reshape(-1).tolist()
        #print(data_exp, data_control)
        if ttest_method == 'simple':
            # use simple t-test
            t, p_value = ttest(data_exp_pooled, data_control_pooled)
        elif ttest_method == 'adjusted': # use adjusted t-test
            t, p_value = adj_ttest(N_per_cluster, N_clusters, inter_day_SD, \
            intra_day_SD, data_exp_pooled, data_control_pooled)
        else:
            print('insert correct t-test method')
    elif data_method == 'cluster':# use means of clusters for processing
        data_exp_mean = data_exp.mean(axis=0)
        data_control_mean = data_control.mean(axis=0)
        if ttest_method == 'simple':
            t, p_value = ttest(data_exp_mean, data_control_mean)
        elif ttest_method == 'adjusted':
            print('can\'t do adjusted t-test. Need pooled data')
            return
        else:
            print('insert correct t-test method')
    return p_value






def experiment(true_exp_value:float, true_control_value:float, \
               inter_day_SD:float, intra_day_SD:float, N_clusters:int, \
               N_per_cluster:int, data_method:str = 'pool', \
               ttest_method:str = 'simple'):
    """ This module generates data and asks another module for processing
    There are several types of processing
    By default it is use simple t-test on pooled data (ignore clustering)
    For more information read documentation for process_data

    INPUT:  1) the parameters for data generating
            2) data_method = {‘pool’, ‘cluster’}, optional
            3) ttest_method = {'simple', 'adjusted'}, optional

    OUTPUT: the p-value of experiment

    EXAMPLE_OF_USE: experiment(1, 1, 0.1, 0.2, 3, 5)
                    experiment(1, 1, 0.1, 0.2, 3, 5, 'cluster', 'adjusted') """
    # generate a matrix of data
    data_exp = generate_data(true_exp_value, inter_day_SD, intra_day_SD, \
                             N_clusters, N_per_cluster)
    data_control = generate_data(true_control_value, inter_day_SD, intra_day_SD,\
                                 N_clusters, N_per_cluster)
    # do the processing
    p_value = process_data(data_exp, data_control, N_per_cluster, \
                                N_clusters, inter_day_SD, intra_day_SD, \
                                data_method, ttest_method)
    # visualize data
    #if show_figure:
        #display_data(data_exp, data_control, N_clusters, N_per_cluster)
    return p_value, data_exp, data_control





def error_probability(NN:int, true_exp_value:float, true_control_value:float, \
                      inter_day_SD:float, intra_day_SD:float, N_clusters:int, \
                      N_per_cluster:int, data_method:str='pool', \
                      ttest_method:str='simple'):
    """ There are two types of errors: 1) False positive 2) False negative
    what are the real values?
    1) In case of unequal initial values we obtain error if p_value > 0.05
       (this means that we agree on zero hypothesis) -> false positive error
    2) If the real values are equal we obtain error if p_value < 0.05
       (thus we reject zero hypothesis) -> false negative error

    INPUT: NN - the number of experiments to conduct
           and other parameters for experiment function

    OUTPUT: the probability of error """
    # sign s will easily help to make < reverse
    if true_exp_value == true_control_value: s = 1
    else: s = -1
    # do NN experiments and see how many times we have an error
    N_error = 0
    for i in range(NN):
        p_value = experiment(true_exp_value, true_control_value, inter_day_SD, \
                             intra_day_SD, N_clusters, N_per_cluster, \
                             data_method, ttest_method)
        if s*p_value < s*0.05 :
            N_error += 1
    return N_error/NN




def error_probability_heatmap(MAX_N_clusters:int, MAX_N_per_cluster:int, \
                              NN:int, true_exp_value:float, \
                              true_control_value:float, inter_day_SD:float, \
                              intra_day_SD:float, N_clusters:int, \
                              N_per_cluster:int, data_method:str='pool', \
                              ttest_method:str='simple'):
    """ Heatmap will show the error probability for an experimentator's choise
    of number of clusters and number of measurements per cluster

    INPUT: MAX_N_clusters - maximum number of clusters (vary from 1 to MAX)
           MAX_N_per_cluster - maximum number of measurements per cluster
           the parameters needed for error_probability function

    OUTPUT: a matrix of probability with axis that correspond to the number
            of clusters and the number od measurements per cluster  """
    CLUSTERS = np.array([i for i in range(1,MAX_N_clusters)])
    PER_CLUSTER = np.array([i for i in range(1,MAX_N_per_cluster)])

    probability = np.zeros((MAX_N_clusters-1, MAX_N_per_cluster-1))
    for i, n_clusters in enumerate(CLUSTERS):
        for j, n_per_cluster in enumerate(PER_CLUSTER):
            probability[i, j] = error_probability(NN, true_exp_value, \
            true_control_value, inter_day_SD, intra_day_SD, n_clusters, \
            n_per_cluster, data_method, ttest_method)
    return probability
    #display_heatmap(probability, CLUSTERS, PER_CLUSTER)




def error_probability_ICC(NN:int, true_exp_value:float, \
                          true_control_value:float, inter_day_SD:float, \
                          intra_day_SD:float, N_clusters:int, \
                          N_per_cluster:int, data_method:str='pool', \
                          ttest_method:str='simple'):
    """ Let's calculate the probability of erroneus result in dependence of ICC
    For this we make the intra_cluster_SD constant and vary inter_cluster_SD
    Then call the function that calculates the probability of error for a
    set of parameters

    INPUT: all the parameters needed for error_probability counting

    OUTPUT: a list of error probability for different ICC & ICC """

    ICC = np.array([0.0, 0.01, 0.03, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, \
                    0.4, 0.45, 0.5])
    inter_day_SDs = np.sqrt(ICC*(intra_day_SD**2)/(1-ICC))

    probability = np.zeros((len(ICC)))
    for i, icc in enumerate(ICC):
        probability[i] = error_probability(NN, true_exp_value, \
                                           true_control_value, inter_day_SDs[i],\
                                           intra_day_SD, N_clusters, \
                                           N_per_cluster, data_method,\
                                           ttest_method)
    return probability, ICC
    #display_graph(probability, ICC)
