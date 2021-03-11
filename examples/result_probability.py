import sys
sys.path.insert(0, '..') # find package 'experiment_simulator' in the previous directory
from scipy.stats import ttest_ind as ttest
from experiment_simulator import *

def result_probability(NN, true_exp_mean, true_control_mean, inter_day_SD, intra_day_SD, N_clusters, N_per_cluster, data_method='pool', ttest_method=True, show_figure=False):
#def result_probability(NN:int, true_exp_mean:float, true_control_mean:float, inter_day_SD:float, intra_day_SD:float, N_clusters:int, N_per_cluster:int, data_method:str = 'pool', ttest_method:bool = True, show_figure:bool = False):
    N_error = 0
    for i in range(NN):
        t, p_value = experiment(true_exp_mean, true_control_mean, inter_day_SD, intra_day_SD, N_clusters, N_per_cluster, 'pool', True, False)
        if p_value < 0.05 :
            N_error += 1

    return N_error/NN

print(experiment(1,1,0.03,0.3,3,50))
print(result_probability(1000,1,1,0.03,0.3,3,50))
