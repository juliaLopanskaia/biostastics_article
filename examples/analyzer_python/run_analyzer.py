import sys
sys.path.insert(0, '../..') # find package in the previous directory
from nested_data_simulator import *


data_exp = read_file('experiment.txt')
data_control = read_file('control.txt')


# Display data
display_data_Superplot(data_exp, data_control)

# Analyze your data
analyze(data_exp, data_control)

