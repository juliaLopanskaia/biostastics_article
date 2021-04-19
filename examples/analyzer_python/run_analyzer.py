import sys
sys.path.insert(0, '../..') # find package in the previous directory
from nested_data_simulator import *

# read data from files in the current directory
data_exp = read_file('experiment.txt')
data_control = read_file('control.txt')

''' uncomment this if you want to manually enter your data
# experimental data:
data_exp =  np.array([
    np.array([0.7,0.8,0.6,0.5,1, 0.8]), #1st cluster
    np.array([1.1,0.9,0.8,0.8,0.9, 1]), #2nd cluster
    np.array([0.7,0.6,0.8,0.9,0.7]) #3rd cluster
], dtype=object)
# control data:
data_control = np.array([
    [1,1,1.5,0.5,1], #1st cluster
    [1.1,1.2,0.8,1.3,1.0, 1.1], #2nd cluster
    [1,1.1,1.3,1.05,0.95]#3rd cluster
], dtype=object)
'''

# Display data
superplot(data_exp, data_control)

# Analyze your data
analyze(data_exp, data_control)
