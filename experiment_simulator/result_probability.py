

def result_probability(NN:int, N_clusters:int, N_per_cluster:int, true_exp_mean:float, true_control_mean:float, inter_day_SD:float, intra_day_SD:float, data_method:bool = True,ttest_method:bool = True):
    """ NN - the number of big experiments (that contain N_clusters and N_per_cluster) """
    N_false = 0
    for i in range(NN):
        experiment()
        if p < 0.05:
            N_false += 1

    return N_false/NN
