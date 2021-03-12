import sys
sys.path.insert(0, '..') # find package in the previous directory
from nested_data_simulator import *
from parameters1 import *


# Simply generate data and display it
data_exp = generate_data(true_exp_value, inter_day_SD, intra_day_SD, \
                             N_clusters, N_per_cluster)
data_control = generate_data(true_control_value, inter_day_SD, intra_day_SD,\
                                 N_clusters, N_per_cluster)
display_data(data_exp, data_control, N_clusters, N_per_cluster)





# Show a dependence or error probability on ICC
[probability, ICC] = error_probability_ICC(NN, true_exp_value, \
                                           true_control_value, inter_day_SD, \
                                           intra_day_SD, N_clusters, \
                                           N_per_cluster, data_method, \
                                           ttest_method)
display_graph(probability, ICC)





# Show heatmap of error plobability in dependence of the number of clusters
# and measurements per cluster 
probability = error_probability_heatmap(MAX_N_clusters, MAX_N_per_cluster, \
                                        NN, true_exp_value, true_control_value,\
                                        inter_day_SD, intra_day_SD, N_clusters,\
                                        N_per_cluster, data_method, \
                                        ttest_method)
