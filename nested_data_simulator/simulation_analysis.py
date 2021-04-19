import numpy as np
from scipy.stats import ttest_ind as ttest
from scipy.stats import t as tpdf
import scipy
from math import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



def generate_data(true_value:float, inter_cluster_SD:float, \
                  intra_cluster_SD:float, N_clusters:int, N_per_cluster:int):
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
    data_exp = generate_data(true_exp_value, inter_cluster_SD, \
                             intra_cluster_SD, N_clusters, N_per_cluster)
    data_control = generate_data(true_control_value, inter_cluster_SD, \
                                 intra_cluster_SD, N_clusters, N_per_cluster)
    # do the processing
    p_value = process_data(data_exp, data_control, inter_cluster_SD, \
                           intra_cluster_SD, data_method, ttest_method)
    return p_value




def error_probability(NN:int, true_exp_value:float, true_control_value:float, \
                      inter_cluster_SD:float, intra_cluster_SD:float, \
                      N_clusters:int, N_per_cluster:int, \
                      data_method:str='pool', ttest_method:str='simple'):
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
        p_value = experiment(true_exp_value, true_control_value, \
                             inter_cluster_SD, intra_cluster_SD, N_clusters, \
                             N_per_cluster, data_method, ttest_method)
        if s*p_value < s*0.05 :
            N_error += 1
    return N_error/NN




def error_probability_heatmap(MAX_N_clusters:int, MAX_N_per_cluster:int, \
                              NN:int, true_exp_value:float, \
                              true_control_value:float, inter_cluster_SD:float,\
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
            true_control_value, inter_cluster_SD, intra_cluster_SD, n_clusters,\
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
                                           true_control_value, \
                                           inter_cluster_SDs[i],\
                                           intra_cluster_SD, N_clusters, \
                                           N_per_cluster, data_method,\
                                           ttest_method)
    return probability




def analyze(data_exp, data_control, forecast_error=False, NN=1000,
            plus_number_cluster=1):
    """
    Analyze your data and print value: Means, standart deviation and p value
    adjusted
    Forecast probability of false negative error if your mean and SD are true
    INPUT:  experimental data (matrix) & control data (matrix)
            forecast_error : bool - Do the forecast probability of false
            negative error?
            NN - number of trials to calculate the probability
            plus_number_cluster - step of increasing the number of clusters
            in the forecast error
    OUTPUT: None """

    inter_cluster_SD, intra_cluster_SD = standard_deviation(data_exp,
                                                            data_control)
    print('Mean experimetal = ', round(pooled(data_exp).mean(),3))
    print('Mean control = ', round(pooled(data_control).mean(), 3))
    print()
    print('inter cluster SD =', round(inter_cluster_SD, 3))
    print('intra cluster SD =', round(intra_cluster_SD, 3))
    ICC = icc_calculator(data_exp, data_control)
    print('ICC = ', round(ICC, 3), '\n')

    if scipy.stats.normaltest(pooled(data_exp))[1] < 0.05:
        print('WARNING: The experimental data are not normally distributed',
              '(with a significance level of 0.05)', '\n',
              'Further results may be incorrect')
    if scipy.stats.normaltest(pooled(data_control))[1] < 0.05:
        print('WARNING: The control data are not normally distributed',
              '(with a significance level of 0.05)', '\n',
              'Further results may be incorrect')

    if len(data_exp) == len(data_control):
        N_clusters = len(data_exp)
    else:
        print('WARNING: the number of clusters for control and experiment are different')
        N_clusters  =  len(data_exp) + len(data_control)

    N_per_cluster = 0
    for data in data_exp:
        N_per_cluster += len(data)
    for data in data_control:
        N_per_cluster += len(data)
    N_per_cluster = N_per_cluster/(len(data_exp)+len(data_control))

    for data in data_exp:
        if abs((len(data) - N_per_cluster)/ N_per_cluster) > 0.2:
            print('WARNING: N per cluster (experiment) : one value is 20% more',
                  'than the average. Further calculations may give inaccurate',
                  'result')
    for data in data_control:
        if abs((len(data) - N_per_cluster)/ N_per_cluster) > 0.2:
            print('WARNING: N per cluster (control): one value is 20% more ',
                  'than the average. Further calculations may give inaccurate',
                  'result.')

    p_value = adj_ttest(N_per_cluster, N_clusters, inter_cluster_SD, \
                        intra_cluster_SD, pooled(data_exp), \
                        pooled(data_control))[1]
    print('p value adjusted = ', round(p_value,3))
    if p_value < 0.05:
        print('Reject the null hypothesis of equality of means with a ',
              'significance level of 0.05')
    else:
        print('There is no reason to reject the null hypothesis of equality',
              'of means with a significance level of 0.05')

    print('\n')
    pb_err = error_probability(NN, pooled(data_exp).mean(), \
                               pooled(data_control).mean(), \
                               inter_cluster_SD, intra_cluster_SD,\
                               len(data_exp), len(data_exp[0]), 'pool', \
                               'adjusted')

    print('Let\'s assume that the measured parameters are true and simulate',
          'data based on them.')

    print('So, the probability of false negative error is', pb_err)

    if forecast_error :
        k = plus_number_cluster
        while pb_err > 0.2:
            pb_err = error_probability(NN, pooled(data_exp).mean(), \
                                       pooled(data_control).mean(), \
                                       inter_cluster_SD, intra_cluster_SD,\
                                       (k + len(data_exp)), \
                                       len(data_exp[0]), 'pool', 'adjusted')
            print('If there were ', k + len(data_exp), ' clusters, the false ',
                  'negative error would be' , pb_err)
            k += plus_number_cluster

    pb_err = error_probability(NN, (pooled(data_exp).mean()+
                               pooled(data_control).mean())/2, \
                               (pooled(data_exp).mean()+
                               pooled(data_control).mean())/2, \
                               inter_cluster_SD, intra_cluster_SD, \
                               len(data_exp), len(data_exp[0]), 'pool', \
                               'adjusted')

    print('\n')
    print('Let\'s assume that your SD are true and means are equal: \n ',
          'mean_exp = mean_control = (mean_exp+ mean_control)/2 = ', \
          round((pooled(data_exp).mean()+pooled(data_control).mean())/2 ,3))

    print('So, the probability of false positive error is', pb_err)




def superplot(data_exp,data_control):
    """
    display data (all experiments and means per clusters) using Superplot
    INPUT: experimental data (matrix) & control data (matrix)
    OUTPUT: None
    """
    #Сreate dataframe from our data
    df = create_dataframe(data_exp,data_control)
    df_mean = create_dataframe(means(data_exp),means(data_control))

    #Plotting swarmplot for all points
    ax = sns.swarmplot(x='control or experiment', y='Arbitrary unit', hue="Ni",\
                       data=df, size=4,alpha=0.5, zorder=1, \
                       palette=sns.color_palette("Spectral", \
                       n_colors=len(df_mean)))

    #Plotting swarmplot for cluster averages
    ax = sns.swarmplot(x="control or experiment", y="Arbitrary unit", hue="Ni",\
                       size=10, edgecolor="k", linewidth=1, data=df_mean, \
                       alpha=1, zorder=2, \
                       palette=sns.color_palette("Spectral", \
                                                 n_colors=len(df_mean)))

    #Build the standard deviation from the mean values for the clusters
    ax.errorbar([0, 1], [means(data_exp).mean(), means(data_control).mean()], \
                xerr=[0.2, 0.2], color='black', elinewidth=2, \
                linewidth=0, zorder=3,  capsize=0)
    ax.errorbar([0, 1], [means(data_exp).mean(), means(data_control).mean()], \
                yerr= [means(data_exp).std()/(np.sqrt(len(data_exp))), \
                means(data_control).std()/(np.sqrt(len(data_control)))], \
                color='black', elinewidth=2, linewidth=0, zorder=3,  capsize=4)

    ax.get_legend().remove()#Removing the legend
    plt.xlim(-0.7,1.7)#Setting the limits on the abscissa axis
    plt.xlabel('')#X-axis signature
    ax.patch.set_visible(False) #Invisible background
    ax.spines['right'].set_visible(False) #Invisible right line drawing boxing
    ax.spines['top'].set_visible(False)#Invisible top line drawing boxing
    plt.show()




def icc_calculator(data_exp,data_control):
    # calculate ICC of your experiment
    inter_cluster_SD, intra_cluster_SD = standard_deviation(data_exp, \
                                                            data_control)
    ICC = inter_cluster_SD**2/(inter_cluster_SD**2 + intra_cluster_SD**2)
    return ICC



def standard_deviation(data_exp,data_control):
    """ Calculate inter cluster and intra cluster standard deviation based on
        experimental data
        INPUT: experimental data (matrix) & control data (matrix)
        OUTPUT: inter cluster SD, intra cluster SD """
    inter_cluster_SD = s_inter(data_exp, data_control)
    intra_cluster_SD = s_intra(data_exp, data_control)
    return inter_cluster_SD, intra_cluster_SD





def means(data):
    """
    INPUT:  data (matrix)
    OUTPUT: means data of clusters
    """
    means = []
    for x in data:
        means.append(np.array(x).mean())
    return np.array(means)




def pooled(data):
    """
    INPUT:  data (matrix)
    OUTPUT: data reshape into (1,n) - line
    """
    pooled = []
    for x in data:
        try :
            for y in x:
                pooled.append(y)
        except:
            pooled.append(x)
    return np.array(pooled)




def s_inter(data_exp, data_control):
    """
    Calculate the standard deviation inter cluster based on experimental data
    INPUT: experimental data (matrix) & control data (matrix)
    OUTPUT: inter cluster SD
    """
    s2 = 0
    mean_exp = pooled(data_exp).mean()
    mean_control = pooled(data_control).mean()
    for x in means(data_exp):
        s2 += (x - mean_exp)**2
    for y in means(data_control):
        s2 += (y  - mean_control)**2
    s2 = s2 / (len(data_exp) + len(data_control) - 2)
    return np.sqrt(s2)




def s_intra(data_exp, data_control):
    """
    Calculate the standard deviation intra cluster based on experimental data
    INPUT: experimental data (matrix) & control data (matrix)
    OUTPUT:  intra cluster SD
    """
    s2 = 0
    for x in data_exp:
        y = np.array(x)
        mean = y.mean()
        for i in y:
            s2 += (i - mean)**2
    for x in data_control:
        y = np.array(x)
        mean = y.mean()
        for i in y:
            s2 += (i - mean)**2
    s2  = s2 / (len(pooled(data_exp)) + len(pooled(data_control)) - len(data_exp) - len(data_control))
    return np.sqrt(s2)




def read_file(file:str):
    matrix = []
    with open(file, 'r') as f:
        m = [[float(num) for num in line.split()] for line in f]
    matrix = np.array(m)
    return matrix




def read_file_csv(file:str):
    matrix = []
    with open(file, 'r') as f:
        for line in f:
            cluster = []
            for num in line.split(';'):
                try:
                    x = float(num)
                    cluster.append(x)
                except:
                    pass
            matrix.append(cluster)
        #m = [[float(num) for num in line.split(';')] for line in f]
    matrix = np.array(matrix)
    return matrix




def create_dataframe(data_exp,data_control):
    """
    Create dataframe from experimental data (matrix) & control data (matrix)
    INPUT: experimental data (matrix) & control data (matrix)
    OUTPUT: dataframe based on experimental data and control data
            dataframe['control or experiment'] - value from control data or experiment data
            dataframe['Arbitrary unit'] - this value
            dataframe['Ni'] - cluster number using end-to-end cluster numbering from experimental and control data
    """
    #Checking whether it is represented by a two-dimensional matrix, or otherwise - a one-dimensional matrix.
    #Depending on this, we set the size of the data. This is for an experimental data
    if type(data_exp[0]) == np.ndarray or type(data_exp[0]) == list:
        N_clusters_exp = len(data_exp)
        N_per_cluster_exp = len(data_exp[0])
        Ni_exp = []
        for Ni, cluster in enumerate(data_exp):
            for _ in cluster:
                Ni_exp.append(Ni)
        Ni_exp = np.array(Ni_exp)
    else:
        N_clusters_exp = 1
        N_per_cluster_exp = len(data_exp)
        Ni_exp = np.array([i for i in range(len(data_exp))])

    #Likewise for control data
    if type(data_control[0]) == np.ndarray or type(data_control[0]) == list:
        N_clusters_control = len(data_control)
        N_per_cluster_control = len(data_control[0])
        Ni_control = []
        for Ni, cluster in enumerate(data_control):
            for _ in cluster:
                Ni_control.append(Ni+1+max(Ni_exp))
        Ni_control = np.array(Ni_control)
    else:
        N_clusters_control = 1
        N_per_cluster_control = len(data_control)
        Ni_control = np.array([(i)+1+max(Ni_exp) for i in range(len(data_control))])

    #Create a list with data belonging to a specific group
    x_exp = np.array(['experiment' for i in range(len(pooled(data_exp)))])
    x_control = np.array(['control' for i in range(len(pooled(data_control)))])

    #create a dictionary corresponding to the future dataframe
    d = {'control or experiment': np.concatenate((x_exp,x_control), axis=0),
         'Arbitrary unit' : np.concatenate((pooled(data_exp), \
                                            pooled(data_control)), axis=0),
         'Ni': np.concatenate((Ni_exp,Ni_control), axis=0)}
    #Create dataframe
    df = pd.DataFrame(data=d)
    return df
