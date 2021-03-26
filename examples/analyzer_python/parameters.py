import numpy as np

N_clusters = 6 # number of clusters
N_per_cluster = 10 # number of experiments per cluster


true_exp_value = 1 # a true value of experimental measurement (e.g. measurement\
                   # with drug)
true_control_value = 1 # a true value of control measurement (e.g. measurement\
                       # without drug)


intra_cluster_SD = 0.3 # the intra day data varience
inter_cluster_SD = 0.15 # the inter day data varience


data_method = 'pool' # either 'pool' or 'cluster'
ttest_method = 'adjusted' # either 'simple' or 'adjusted'


NN = 100 # the number of experiments to conduct (for error probability \
         # counting) - optional
MAX_N_clusters = 11 # maximum number of clusters (for heatmap) - optional
MAX_N_per_cluster = 21 # maximum number of measurements per cluster to check \
                       #(for heatmap) - optional

ICC = np.array([0.0, 0.01, 0.03, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, \
                    0.4, 0.45, 0.5]) # array for error probability dependence
                    # (optional) - ICC to use for calculation
