""" The package has tools to simulate an experiment.
    Module generate_data can generate data
    figure_display displays all the data in a figure
    ttest_print calculates ttest based on the data and prints out the result
    adj_ttest_print calculates adjusted ttest and prints out the result"""

#__all__ = ['generate_data', 'ttest_print', 'figure_generate', 'correct_p_value']
from experiment_simulator.generate_data import generate_data
from experiment_simulator.figure_display import figure_display
from experiment_simulator.ttest_print import ttest_print
from experiment_simulator.adj_ttest_print import adj_ttest_print
