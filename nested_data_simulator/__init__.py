""" The package has tools to simulate an experiment.
    Module generate_data can generate data
    figure_display displays all the data in a figure
    ttest_print calculates ttest based on the data and prints out the result
    adj_ttest_print calculates adjusted ttest and prints out the result"""

__all__ = ['generate_data', 'adj_ttest', 'display_data']
#__all__ = ['generate_data', 'adj_ttest', 'figure_display']
from nested_data_simulator.generate_data import generate_data
#from nested_data_simulator.figure_display import figure_display
#from experiment_simulator.ttest_print import ttest_print
from nested_data_simulator.adj_ttest import adj_ttest
#from nested_data_simulator.experiment import experiment
#from experiment_simulator.maxim import maxim
from nested_data_simulator.display_data import display_data
