import sys
sys.path.insert(0, '..') # find package in the previous directory
from nested_data_simulator import *
from parameters import *


#Your experimental data
#Example:
data_exp =  np.array([
    [0.7,0.8,0.6,0.5,1], #1st claster
    [1.1,0.9,0.8,0.8,0.9], #2nd claster
    [0.7,0.6,0.8,0.9,0.7] #3rd claster
])
#control or comparing data
#Example:
data_control = np.array([
    [1,1,1.5,0.5,1], #1st claster
    [1.1,1.2,0.8,1.3,1.0], #2nd claster
    [0.8,0.9,1,0.7,0.8] #3rd claster
])

# Display data
display_data_Superplot(data_exp, data_control)

# Analyze your data
analyze(data_exp, data_control)

