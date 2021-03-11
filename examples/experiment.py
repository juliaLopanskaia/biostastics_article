import sys
sys.path.insert(0, '..') # find package 'experiment_simulator' in the previous directory
from nested_data_simulator import *
#from nested_data_simulator.display_data import display_data
#from numpy import *
from scipy.stats import ttest_ind as ttest
from parameters1 import *

def experiment(true_exp_mean:float, true_control_mean:float, inter_day_SD:float, intra_day_SD:float, N_clusters:int, N_per_cluster:int, data_method:str = 'pool', ttest_method:bool = True, show_figure:bool = True):
    """ This module generates data and does the processing
        There are several types of processing
        By default it is use simple t-test on pooled data (ignore clustering)

    INPUT:  1) the parameters for data generating
            2) data_method = {‘pool’, ‘cluster_means’}, optional
                   choose the type of data to process furter
                   ( if 'pool', use the pooled data
                     elif 'cluster_means' use the means of clusters )
            2) ttest_method: bool, optional
                   choose what type of ttest to apply
                   ( if True, use simple t-test
                     else use the adjusted t-test )
            3) figure_show: bool, optional
                   decide if you want to see the figure of your data
                   by default it's off

    OUTPUT: hypothesis and p-value of experiment result

    EXAMPLE_OF_USE: experiment(1, 1, 0.1, 0.2, 3, 5) """

    # generate a matrix of data
    data_exp = generate_data(true_exp_mean, inter_day_SD, intra_day_SD, N_clusters, N_per_cluster)
    data_control = generate_data(true_control_mean, inter_day_SD, intra_day_SD, N_clusters, N_per_cluster)
    # do the processing
    #process_data()      FIXME
    #ipdb.set_trace()
    if data_method == 'pool': # use pooled data for processing
        data_exp_pooled = data_exp.reshape(-1).tolist() # pool the data into a list
        data_control_pooled = data_control.reshape(-1).tolist()
        #print(data_exp, data_control)
        if ttest_method:
            t, p_value = ttest(data_exp_pooled, data_control_pooled) # use simple t-test
        else: # use adjusted t-test
            t, p_value = adj_ttest(N_per_cluster, N_clusters, inter_day_SD, intra_day_SD, data_exp_pooled, data_control_pooled)
    elif data_method == 'cluster_means':# use means of clusters for processing
        data_exp_mean = data_exp.mean(axis=0)
        data_control_mean = data_control.mean(axis=0)
        if ttest_method:
            t, p_value = ttest(data_exp_mean, data_control_mean) # calculate t-test and check a hypothesis
        else:
            print('can\'t do adjusted t-test on means of clusters. Need pooled data')
            return
    # display data
    #ipdb.set_trace()
    if show_figure:
        #maxim()
        mean_exp = data_exp.mean(axis=0)
        mean_control = data_control.mean(axis=0)
        #figure_display(N_clusters, N_per_cluster, data_exp, mean_exp, data_control, mean_control)
        display_data(data_exp, data_control, N_clusters, N_per_cluster)

    return t, p_value

experiment(true_exp_mean, true_control_mean, inter_day_SD, intra_day_SD, N_clusters, N_per_cluster)
