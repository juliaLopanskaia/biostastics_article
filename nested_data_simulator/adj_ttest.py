import numpy as np
from scipy.stats import t as tpdf
from math import *

def adj_ttest(N_per_cluster, N_clusters, inter_day_SD, intra_day_SD, data_exp_pooled, data_control_pooled):
    N = N_per_cluster*N_clusters # total number of experiments
    ICC = inter_day_SD**2/(inter_day_SD**2 + intra_day_SD**2); # intraclass correlation calculation
    c = sqrt(((N - 2) - 2*(N_per_cluster-1)*ICC)/((N-2)*(1+(N_per_cluster-1)*ICC))) # correction factor for t-distribution     FIXME rename as items
    df = ((N-2)-2*(N_per_cluster-1)*ICC)**2/((N-2)*(1-ICC)**2+N_per_cluster-1*(N-2*N_per_cluster-1)*ICC+2*(N-2*N_per_cluster)*ICC*(1-ICC)) # corrected degrees of freedom
    s = sqrt(((N-1)*np.std(data_exp_pooled)**2+(N-1)*np.std(data_control_pooled)**2)/(2*N-2)) # standard deviation of two datasets
    t = abs(np.mean(data_exp_pooled) - np.mean(data_control_pooled))/(s*sqrt(1/N + 1/N)) # t-test
    ta = c*t # corrected t-test
    p_value = 2*sum(tpdf.pdf(np.arange(ta,100,0.001),df)*0.001) # p-value = integral of t-distribution probability function
    #print('P-value based on t-distribution probability function is {:2.2f}'.format(p_value))
    h = 1 # FIXME
    return h, p_value
