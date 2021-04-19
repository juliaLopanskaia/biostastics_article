""" The package has tools to simulate an experiment.
    Module generate_data can generate data
    figure_display displays all the data in a figure
    ttest_print calculates ttest based on the data and prints out the result
    adj_ttest_print calculates adjusted ttest and prints out the result"""

__all__ = ['generate_data', 'adj_ttest', 'process_data', 'experiment', \
           'error_probability', 'error_probability_heatmap', 'icc_calculator',
           'error_probability_ICC', 'display_heatmap', 'display_graph', \
           'superplot', 'analyze', 'read_file', 'read_file_csv']

from nested_data_simulator.simulation_analysis import generate_data
from nested_data_simulator.simulation_analysis import adj_ttest
from nested_data_simulator.simulation_analysis import process_data
from nested_data_simulator.simulation_analysis import experiment
from nested_data_simulator.simulation_analysis import error_probability
from nested_data_simulator.simulation_analysis import error_probability_heatmap
from nested_data_simulator.simulation_analysis import error_probability_ICC
from nested_data_simulator.simulation_analysis import analyze
from nested_data_simulator.simulation_analysis import icc_calculator
from nested_data_simulator.simulation_analysis import read_file
from nested_data_simulator.simulation_analysis import read_file_csv
from nested_data_simulator.simulation_analysis import superplot
from nested_data_simulator.display import display_graph
from nested_data_simulator.display import display_heatmap
