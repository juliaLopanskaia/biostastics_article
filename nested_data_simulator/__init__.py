""" The package has tools to simulate an experiment.
    Module generate_data can generate data
    figure_display displays all the data in a figure
    ttest_print calculates ttest based on the data and prints out the result
    adj_ttest_print calculates adjusted ttest and prints out the result"""

__all__ = ['generate_data', 'adj_ttest', 'process_data', 'experiment', \
           'error_probability', 'error_probability_heatmap', \
           'error_probability_ICC', 'display_data', 'display_heatmap', \
           'display_graph']
#__all__ = ['generate_data', 'adj_ttest', 'figure_display']
from nested_data_simulator.simulator import generate_data
from nested_data_simulator.simulator import adj_ttest
from nested_data_simulator.simulator import process_data
from nested_data_simulator.simulator import experiment
from nested_data_simulator.simulator import error_probability
from nested_data_simulator.simulator import error_probability_heatmap
from nested_data_simulator.simulator import error_probability_ICC
from nested_data_simulator.display import display_data
from nested_data_simulator.display import display_graph
from nested_data_simulator.display import display_heatmap
