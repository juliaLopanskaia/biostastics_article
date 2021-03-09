# simulation to understand biostatistics
import sys
sys.path.insert(0, '..') # this helps to find package 'experiment_simulator' in the previous directory
from experiment_simulator import *
import numpy as np
from parameters import *

#experiment_simulator.generate_data()
data_exp, mean_exp, data_exp_pooled = generate_data(true_exp_mean, inter_day_SD, intra_day_SD, N_clusters, N_per_cluster)
data_control, mean_control, data_control_pooled = generate_data(true_control_mean, inter_day_SD, intra_day_SD, N_clusters, N_per_cluster)

# draw a figure of data (means of clusters are shown with a big plus sign)
figure_generate(N_per_cluster, N_clusters, data_exp, mean_exp, data_control, mean_control) # FIXME

# Three types of data processing. We fo all three to understand how they afect the result
ttest_print(data_exp_pooled, data_control_pooled, '(pooled data)') # t-test based on pooled data
ttest_print(mean_exp, mean_control, '(per-day means)') # t-test based on per day means (means of clusters)
correct_p_value(N_per_cluster, N_clusters, inter_day_SD, intra_day_SD, data_exp_pooled, data_control_pooled) # calculation of correct p-value
# print out the measured values
print('Mean value in experiment {:2.2f}'.format(np.mean(mean_exp)))
print('Mean value in control {:2.2f} \n'.format(np.mean(mean_control)))
