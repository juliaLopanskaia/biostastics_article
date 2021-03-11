# simulation to understand biostatistics
import sys
sys.path.insert(0, '..') # find package 'experiment_simulator' in the previous directory
from nested_data_simulator import *
import numpy as np
from parameters1 import *

def test(show_figure:bool=True)
# generate a matrix of measured values in experiment
    data_exp = generate_data(true_exp_mean, inter_day_SD, intra_day_SD, N_clusters, N_per_cluster)
    data_control = generate_data(true_control_mean, inter_day_SD, intra_day_SD, N_clusters, N_per_cluster)
#data_exp = np.matrix([[1, 1.1], [1.2, 1.3]])
#data_control = np.matrix([[1, 1.1], [1.2, 1.3]])
# draw a figure of data (means of clusters are shown with big plus sign)
    data_exp_pooled = data_exp.reshape(-1).tolist() # pool the data into a list
    data_control_pooled = data_control.reshape(-1).tolist()
    t, p_value = adj_ttest(N_per_cluster, N_clusters, inter_day_SD, intra_day_SD, data_exp_pooled, data_control_pooled)


    mean_exp = data_exp.mean(axis=0)
    mean_control = data_control.mean(axis=0)
    figure_display(N_clusters, N_per_cluster, data_exp, mean_exp, data_control, mean_control)
    if show_figure:
        #maxim()
        mean_exp = data_exp.mean(axis=0)
        mean_control = data_control.mean(axis=0)
        figure_display(N_clusters, N_per_cluster, data_exp, mean_exp, data_control, mean_control)

test()
#display_data(data_exp, data_control, N_clusters, N_per_cluster) # FIXME


#import sys
#sys.path.insert(0, '..') # find package 'experiment_simulator' in the previous directory
#from experiment_simulator import *
#import numpy as np
#from parameters1 import *

# generate a matrix of measured values in experiment
#data_exp, mean_exp, data_exp_pooled = generate_data(true_exp_mean, inter_day_SD, intra_day_SD, N_clusters, N_per_cluster)
#data_control, mean_control, data_control_pooled = generate_data(true_control_mean, inter_day_SD, intra_day_SD, N_clusters, N_per_cluster)

# draw a figure of data (means of clusters are shown with big plus sign)
#figure_display(N_per_cluster, N_clusters, data_exp, mean_exp, data_control, mean_control) # FIXME

# Three types of data processing. We do all three to understand how they affect the result
#ttest_print(data_exp_pooled, data_control_pooled, '(pooled data)') # t-test based on pooled data
#ttest_print(mean_exp, mean_control, '(per-day means)') # t-test based on per day means (means of clusters)
#adj_ttest_print(N_per_cluster, N_clusters, inter_day_SD, intra_day_SD, data_exp_pooled, data_control_pooled) # calculation of correct p-value
# print out the measured values
#print('Mean value in experiment {:2.2f}'.format(np.mean(mean_exp)))
#print('Mean value in control {:2.2f} \n'.format(np.mean(mean_control)))
