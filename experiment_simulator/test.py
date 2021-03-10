from numpy import *
import sys
sys.path.insert(0, '..') # find package 'experiment_simulator' in the previous directory
from experiment_simulator import *
def test():
    #data = generate_data(1, 0.1, 0.1, 3, 5)
    #mean_cluster = data.mean(axis=0);
    #data = data.reshape(-1).tolist()
    #print(data)
    #print(data_pooled)
    #print(mean_cluster)
    t, p_value = experiment(1, 1, 0.01, 0.02, 3, 50, 'pool', True, True)
    print(t, p_value)
test()
