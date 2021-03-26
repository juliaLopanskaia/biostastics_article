import numpy as np
from scipy.stats import ttest_ind as ttest
from scipy.stats import t as tpdf
import scipy
from math import *
import seaborn as sns




def read_file(file:str):
    matrix = []
    with open(file, 'r') as f:
        m = [[float(num) for num in line.split()] for line in f]
    matrix = np.array(m)
    return matrix




def generate_data(true_value:float, inter_cluster_SD:float, intra_cluster_SD:float, \
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
    data = true_value + inter_cluster_SD*np.random.randn(1,N_per_cluster) + \
           intra_cluster_SD*np.random.randn(N_clusters,N_per_cluster)
    return data




def adj_ttest(N_per_cluster:int, N_clusters:int, inter_cluster_SD:float, \
              intra_cluster_SD:float, data_exp_pooled:list, \
              data_control_pooled:list):
    N = N_per_cluster*N_clusters # the total number of experiments
    # calculate intraclass correlation calculation:
    ICC = inter_cluster_SD**2/(inter_cluster_SD**2 + intra_cluster_SD**2);

    item1 = (N_per_cluster - 1)*ICC
    item2 = (N - 2) - 2*item1
    item3 = (N - 2)*(1 + item1)
    c = np.sqrt(item2/item3) # correction factor for t-distribution

    item4 = N-2*N_per_cluster
    item5 = (N-2)*(1-ICC)**2
    item6 = N_per_cluster*item4*ICC**2
    item7 = 2*item4*ICC*(1 - ICC)
    h = item2**2/(item5 + item6 + item7) # corrected degrees of freedom

    # standard deviation of two datasets:
    s=np.sqrt((N*data_exp_pooled.std()**2+N*data_control_pooled.std()**2)/(2*N-2))

    # t-test
    t = abs(np.mean(data_exp_pooled) - np.mean(data_control_pooled))/(s*np.sqrt(1/N + 1/N)) 
    ta = c*t # corrected t-test
    # p-value = integral of t-distribution probability function:
    p_value = 2*(1-tpdf.cdf(ta, h))
    #print('P-value based on t-distribution probability function is {:2.2f}'.format(p_value))
    return ta, p_value




def process_data(data_exp, data_control, inter_cluster_SD, intra_cluster_SD,\
		  data_method, ttest_method):
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
    N_clusters = len(data_exp)
    N_per_cluster = len(data_exp[0])
    if data_method == 'pool': # use pooled data for processing
        # pool the data into a list:
        data_exp_pooled = data_exp.reshape(-1)
        data_control_pooled = data_control.reshape(-1)
        #print(data_exp, data_control)
        if ttest_method == 'simple':
            # use simple t-test
            t, p_value = ttest(data_exp_pooled, data_control_pooled)
        elif ttest_method == 'adjusted': # use adjusted t-test
            t, p_value = adj_ttest(N_per_cluster, N_clusters, inter_cluster_SD, \
            intra_cluster_SD, data_exp_pooled, data_control_pooled)
        else:
            print('insert correct t-test method')
    elif data_method == 'cluster':# use means of clusters for processing
        data_exp_mean = data_exp.mean(axis=1)
        data_control_mean = data_control.mean(axis=1)
        if ttest_method == 'simple':
            t, p_value = ttest(data_exp_mean, data_control_mean)
        elif ttest_method == 'adjusted':
            print('can\'t do adjusted t-test. Need pooled data')
            return
        else:
            print('insert correct t-test method')
    else: print('what kind of data_method do you use?')
    return p_value




def experiment(true_exp_value:float, true_control_value:float, \
               inter_cluster_SD:float, intra_cluster_SD:float, N_clusters:int, \
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
    # generate 2 matrices of data (control and experiment)
    data_exp = generate_data(true_exp_value, inter_cluster_SD, intra_cluster_SD, \
                             N_clusters, N_per_cluster)
    data_control = generate_data(true_control_value, inter_cluster_SD, \
                                 intra_cluster_SD, N_clusters, N_per_cluster)
    # do the processing
    p_value = process_data(data_exp, data_control, inter_cluster_SD, intra_cluster_SD,\
                           data_method, ttest_method)
    return p_value




def error_probability(NN:int, true_exp_value:float, true_control_value:float, \
                      inter_cluster_SD:float, intra_cluster_SD:float, N_clusters:int, \
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
        p_value = experiment(true_exp_value, true_control_value, inter_cluster_SD,\
                             intra_cluster_SD, N_clusters, N_per_cluster, \
                             data_method, ttest_method)
        if s*p_value < s*0.05 :
            N_error += 1
    return N_error/NN




def error_probability_heatmap(MAX_N_clusters:int, MAX_N_per_cluster:int, \
                              NN:int, true_exp_value:float, \
                              true_control_value:float, inter_cluster_SD:float, \
                              intra_cluster_SD:float, data_method:str='pool', \
                              ttest_method:str='simple'):
    """ Heatmap will show the error probability for an experimentator's choise
    of number of clusters and number of measurements per cluster

    INPUT: MAX_N_clusters - maximum number of clusters (vary from 1 to MAX)
           MAX_N_per_cluster - maximum number of measurements per cluster
           the parameters needed for error_probability function

    OUTPUT: a matrix of probability with axis that correspond to the number
            of clusters and the number od measurements per cluster  """
    CLUSTERS = np.array([i for i in range(2,MAX_N_clusters+1)])
    PER_CLUSTER = np.array([i for i in range(2,MAX_N_per_cluster+1)])

    probability = np.zeros((MAX_N_clusters-1, MAX_N_per_cluster-1))
    for i, n_clusters in enumerate(CLUSTERS):
        for j, n_per_cluster in enumerate(PER_CLUSTER):
            probability[i, j] = error_probability(NN,true_exp_value, \
            true_control_value, inter_cluster_SD, intra_cluster_SD, n_clusters, \
            n_per_cluster, data_method, ttest_method)
    return probability




def error_probability_ICC(NN:int, true_exp_value:float, \
                          true_control_value:float, \
                          intra_cluster_SD:float, N_clusters:int, \
                          N_per_cluster:int, ICC, \
                          data_method:str='pool', ttest_method:str='simple'):
    """ Let's calculate the probability of erroneus result in dependence of ICC
    For this we make the intra_cluster_SD constant and vary inter_cluster_SD
    Then call the function that calculates the probability of error for a
    set of parameters

    INPUT: all the parameters needed for error_probability counting

    OUTPUT: a list of error probability for different ICC & ICC """

    inter_cluster_SDs = np.sqrt(ICC*(intra_cluster_SD**2)/(1-ICC))

    probability = np.zeros((len(ICC)))
    for i in range(len(ICC)):
        probability[i] = error_probability(NN, true_exp_value, \
                                           true_control_value, inter_cluster_SDs[i],\
                                           intra_cluster_SD, N_clusters, \
                                           N_per_cluster, data_method,\
                                           ttest_method)
    return probability
    
    

    
def standard_deviation(data_exp,data_control):
    """
    Calculate the standard deviation inter cluster and intra cluster 
    based on experimental data
    INPUT: experimental data (matrix) & control data (matrix)
    OUTPUT: inter cluster SD, intra cluster SD
    """
    inter_cluster_SD = data_exp.mean(axis=1).std(ddof=1)
    data_std = data_exp.std(axis=1, ddof=1)
    intra_cluster_SD = np.sqrt((data_std**2).sum()/(len(data_exp)))
    return inter_cluster_SD, intra_cluster_SD




def analyze(data_exp, data_control, forecast_error=True, NN=1000, plus_number_cluster=1):
    """
    Analyze your data and print value: Means, standart deviation and p value adjusted
    Display data Superplot
    Forecast probability of false negative error if your mean and SD are true
    INPUT:  experimental data (matrix) & control data (matrix)
            forecast_error : bool - Do the forecast probability of false negative error?
            NN - number of trials to calculate the probability
            plus_number_cluster - step of increasing the number of clusters in the forecast error
    OUTPUT: None
                    
    """
    inter_cluster_SD, intra_cluster_SD = standard_deviation(data_exp, data_control)
    
    print('Mean experimetal = ', round(data_exp.mean(),3))
    print('Mean control = ', round(data_control.mean(), 3))
    print()
    print('inter cluster SD =', round(inter_cluster_SD, 3))
    print('intra cluster SD =', round(intra_cluster_SD, 3))
    ICC = icc_calculator(data_exp, data_control)
    print('ICC = ', round(ICC, 3))
    print('\n')
    p_value = process_data(data_exp, data_control,  \
                          inter_cluster_SD=inter_cluster_SD, intra_cluster_SD=intra_cluster_SD,\
                          data_method='pool', ttest_method='adjusted')
    print('p value adjusted = ', round(p_value,3))
    if p_value < 0.05:
        print('Reject the null hypothesis of equality of means with a significance level of 0.05')
    else:
        print('There is no reason to reject the null hypothesis of equality of means with a significance '\
              'level of 0.05')
                                
    print('\n')
    pb_err = error_probability(NN=NN, true_exp_value=data_exp.mean(), true_control_value=data_control.mean(), \
                          inter_cluster_SD=inter_cluster_SD, intra_cluster_SD=intra_cluster_SD,\
                            N_clusters=len(data_exp), \
                          N_per_cluster=len(data_exp[0]), data_method='pool', \
                          ttest_method='adjusted')

    print('Let\'s analyze the result')
    print('In order to simulate more experiments based on parameters from your data,\nwe have ',\
          'to assume that the measured parameters are true (as they appear to be in nature).')
    print('So, for now we have simulated N =',NN,'experiments and the probability of false negative error is', pb_err)
    
    if forecast_error :
        k = plus_number_cluster
        while pb_err > 0.2:
            pb_err = error_probability(NN=NN, true_exp_value=data_exp.mean(), \
                                       true_control_value=data_control.mean(), \
                                       inter_cluster_SD=inter_cluster_SD, \
                                       intra_cluster_SD=intra_cluster_SD,\
                                       N_clusters=(k + len(data_exp)), \
                                       N_per_cluster=len(data_exp[0]), data_method='pool', \
                                       ttest_method='adjusted')
            print('If there were ', k + len(data_exp), ' clusters, the false negative error would be' , pb_err)
            k += plus_number_cluster
            
    pb_err = error_probability(NN=NN, true_exp_value=(data_exp.mean()+data_control.mean())/2, \
                               true_control_value=(data_exp.mean()+data_control.mean())/2, \
                          inter_cluster_SD=inter_cluster_SD, intra_cluster_SD=intra_cluster_SD,\
                            N_clusters=len(data_exp), \
                          N_per_cluster=len(data_exp[0]), data_method='pool', \
                          ttest_method='adjusted')
    print('\n')
    print('What if the means that you obtained are not true? This can hypothetically '\
          'be the false positive result.')
    print('Let\'s assume that your SD are true and the means of control and experiment are not. '\
          '\nThe means will be equal to (mean_exp+ mean_control)/2 =', \
          round((data_exp.mean()+data_control.mean())/2 ,3))

    print('So, the probability of false positive error is', pb_err)




def icc_calculator(data_exp,data_control):
    
    inter_cluster_SD, intra_cluster_SD = standard_deviation(data_exp, data_control)
    ICC = inter_cluster_SD**2/(inter_cluster_SD**2 + intra_cluster_SD**2)
    
    return ICC 
