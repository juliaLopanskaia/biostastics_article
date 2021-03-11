from scipy.stats import ttest_ind as ttest

def ttest_print(data1:list, data2:list, data_name:str = ''):
    """ This module calculates t-test and prints out the result

    INPUT: two lists of data that need to be compared and if you wish the
    name of data to print out detailed information (it's nothing by default).

    OUTPUT: None """

    t, p_value = ttest(data1, data2); # calculate t-test and check a hypothesis
    if h==0: # if zero hypothesis is correct
        print('Means are the same ' + data_name + ',  p-value is {:2.2f}'.format(p_value))
    else: # if zero hypothesis is incorrect
        print('Means are different ' + data_name + ',  p-value is {:2.2f}'.format(p_value))
    return t, p_value
