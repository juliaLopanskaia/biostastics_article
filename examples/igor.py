import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns; sns.set_theme()
from scipy import stats
from random import randint

def p_values(N_per_day = 10, N_days = 5, true_exp_mean = 1, true_control_mean = 1, inter_day_SD = 0.1, sigma = 0.3, graph=False):
    
    mean_exp = true_exp_mean*(1 + inter_day_SD*np.random.normal(0, 1, N_days))
    mean_control = true_control_mean*(1 + inter_day_SD*np.random.normal(0, 1, N_days))
    day_exp = []
    mean_days_exp = []
    
    day_control = []
    mean_days_control = []
    
    for i in range(N_days):
        tmp = mean_exp[i]*(1 + sigma*np.random.normal(0, 1, N_per_day))
        mean_days_exp.append(tmp.mean())
        day_exp.append(tmp)

        tmp1 = mean_control[i]*(1 + sigma*np.random.normal(0, 1, N_per_day))
        mean_days_control.append(tmp1.mean())
        day_control.append(mean_control[i] + sigma*np.random.normal(0, 1, N_per_day))

    day_exp = np.array(day_exp)
    mean_days_exp = np.array(mean_days_exp)

    day_control = np.array(day_control)
    mean_days_control = np.array(mean_days_control)
    p_value_all = stats.ttest_ind(day_exp.reshape((N_days*N_per_day)), day_control.reshape((N_days*N_per_day)))[1]
    p_value_mean = stats.ttest_ind(mean_days_exp,  mean_days_control)[1]
    
    if graph:
        colors = []

        for i in range(N_days):
            colors.append('#%06X' % randint(0, 0xFFFFFF))
        for i in range(N_days):
            plt.scatter(np.random.uniform(0.9,1.1,N_per_day), day_exp[i], color=colors[i])
            plt.scatter(np.random.uniform(1.9,2.1,N_per_day), day_control[i], color=colors[i])
            plt.axis([0,3,-1,2])
            plt.plot([0.8,1.2], [mean_days_exp[i],mean_days_exp[i]], color=colors[i])
            plt.plot([1.8,2.2], [mean_days_control[i],mean_days_control[i]], color=colors[i])
            plt.xlabel('exp                          control') 
    
    return [p_value_all, p_value_mean]

def probability_false_positive(N=1000, N_per_day = 10, N_days = 5, true_exp_mean = 1, true_control_mean = 1, inter_day_SD = 0.1, sigma = 0.3):
    false_all = 0
    false_mean = 0
    for i in range(N):
        p = p_values(N_per_day = N_per_day, N_days = N_days, true_exp_mean = true_exp_mean, true_control_mean = true_control_mean, inter_day_SD = inter_day_SD, sigma = sigma)
        if p[0] < 0.05 :
            false_all += 1
        if p[1] < 0.05:
            false_mean +=1
            
    return [false_all/N, false_mean/N ]

def probability_false_negative(N=1000, N_per_day = 10, N_days = 5, true_exp_mean = 0.8, true_control_mean = 1, inter_day_SD = 0.1, sigma = 0.3):
    false_all = 0
    false_mean = 0
    for i in range(N):
        p = p_values(N_per_day = N_per_day, N_days = N_days, true_exp_mean = true_exp_mean, true_control_mean = true_control_mean, inter_day_SD = inter_day_SD, sigma = sigma)
        if p[0] > 0.05 :
            false_all += 1
        if p[1] > 0.05:
            false_mean +=1
            
    return [false_all/N, false_mean/N ]


def heatmap_probability_false_positive():
    DAYS = np.array([i for i in range(1,11)])
    PER_DAY = np.array([i for i in range(1,21)])
    Number_DAYS = len(DAYS)
    Numbers_PER_DAY = len(PER_DAY)

    probabilities_all = np.zeros((Number_DAYS, Numbers_PER_DAY))
    probabilities_mean = np.zeros((Number_DAYS, Numbers_PER_DAY))

    for i,days in enumerate(DAYS):
        for j,per_day in enumerate(PER_DAY):
            probably = probability_false_positive(N_per_day=per_day, N_days=days, true_exp_mean = 1, true_control_mean = 1, inter_day_SD = 0.1, sigma = 0.3)
            probabilities_all[i, j] = probably[0] 
            probabilities_mean[i, j] = probably[1]
    ax = sns.heatmap(probabilities_all.T, xticklabels=DAYS, yticklabels=PER_DAY)
    ax.invert_yaxis()
    plt.xlabel('Days')
    plt.ylabel('N_per_day')
    plt.title('All')
    plt.show()

    ax = sns.heatmap(probabilities_mean.T, xticklabels=DAYS, yticklabels=PER_DAY)
    ax.invert_yaxis()
    plt.xlabel('Days')
    plt.ylabel('N_per_day')
    plt.title('Mean')

def heatmap_probability_false_negative():
    DAYS = np.array([i for i in range(1,11)])
    PER_DAY = np.array([i for i in range(1,21)])
    Number_DAYS = len(DAYS)
    Numbers_PER_DAY = len(PER_DAY)

    probabilities_all = np.zeros((Number_DAYS, Numbers_PER_DAY))
    probabilities_mean = np.zeros((Number_DAYS, Numbers_PER_DAY))

    for i,days in enumerate(DAYS):
        for j,per_day in enumerate(PER_DAY):
            probably = probability_false_negative(N_per_day=per_day, N_days=days, true_exp_mean = 0.8, true_control_mean = 1, inter_day_SD = 0.1, sigma = 0.3)
            probabilities_all[i, j] = probably[0] 
            probabilities_mean[i, j] = probably[1]
    ax = sns.heatmap(probabilities_all.T, xticklabels=DAYS, yticklabels=PER_DAY)
    ax.invert_yaxis()
    plt.xlabel('Days')
    plt.ylabel('N_per_day')
    plt.title('All')
    plt.show()

    ax = sns.heatmap(probabilities_mean.T, xticklabels=DAYS, yticklabels=PER_DAY)
    ax.invert_yaxis()
    plt.xlabel('Days')
    plt.ylabel('N_per_day')
    plt.title('Mean')


def probability_false_positive_ICC(graph=True, Sigma2_constant=True):
    ICC = np.array([0.0, 0.01, 0.03, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    
    if Sigma2_constant:
        SIGMA2 = 0.1
        INTER_day_SD = np.sqrt(ICC*SIGMA2)
        Sigmas = np.sqrt(SIGMA2-INTER_day_SD**2)
        
    else:
        Sigmas = 0.3+0*ICC
        INTER_day_SD = np.sqrt(ICC*(Sigmas**2)/(1-ICC))
        

    Number_ICC = len(ICC)


    probabilities_all = np.zeros((Number_ICC))
    probabilities_mean = np.zeros((Number_ICC))



    for i, icc in enumerate(ICC):

        probably = probability_false_positive(N=1000, N_per_day=10, N_days=10, true_exp_mean = 1, true_control_mean = 1, inter_day_SD = INTER_day_SD[i], sigma = Sigmas[i])
        probabilities_all[i] = probably[0] 
        probabilities_mean[i] = probably[1]
            
    if graph:
        fig, ax = plt.subplots()
        ax.scatter(ICC, probabilities_all, label='All')
        ax.scatter(ICC, probabilities_mean, label='Mean')

        #plt.plot(ICC, probabilities_all, label='All')
        #plt.plot(ICC, probabilities_mean, label='Mean')
        ax.legend()
        plt.xlabel('ICC')
        
    return np.array([[ICC, probabilities_all],
                     [ICC, probabilities_mean]])

def probability_false_negative_ICC(graph=True, Sigma2_constant=True):
    ICC = np.array([0.0, 0.01, 0.03, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    
    if Sigma2_constant:
        SIGMA2 = 0.1
        INTER_day_SD = np.sqrt(ICC*SIGMA2)
        Sigmas = np.sqrt(SIGMA2-INTER_day_SD**2)
        
    else:
        Sigmas = 0.3+0*ICC
        INTER_day_SD = np.sqrt(ICC*(Sigmas**2)/(1-ICC))
        

    Number_ICC = len(ICC)


    probabilities_all = np.zeros((Number_ICC))
    probabilities_mean = np.zeros((Number_ICC))

    for i, icc in enumerate(ICC):

        probably = probability_false_negative(N=1000, N_per_day=10, N_days=10, true_exp_mean = 0.8, true_control_mean = 1, inter_day_SD = INTER_day_SD[i], sigma = Sigmas[i])
        probabilities_all[i] = probably[0] 
        probabilities_mean[i] = probably[1]
            
    if graph:
        fig, ax = plt.subplots()
        ax.scatter(ICC, probabilities_all, label='All')
        ax.scatter(ICC, probabilities_mean, label='Mean')

        #plt.plot(ICC, probabilities_all, label='All')
        #plt.plot(ICC, probabilities_mean, label='Mean')
        ax.legend()
        plt.xlabel('ICC')
        
    return np.array([[ICC, probabilities_all],
                     [ICC, probabilities_mean]])








heatmap_probability_false_positive()
heatmap_probability_false_negative()
p_values(graph=True)
probability_false_positive_ICC()
probability_false_negative_ICC()

