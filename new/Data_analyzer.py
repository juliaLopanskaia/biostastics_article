#!/usr/bin/env python
# coding: utf-8


import code


# ### Analyzing the data that we enter here


#Your experimental data
#Example:
data_exp =  np.array([
    np.array([0.7,0.8,0.6,0.5,1, 0.8]), #1st claster
    np.array([1.1,0.9,0.8,0.8,0.9, 1]), #2nd claster
    np.array([0.7,0.6,0.8,0.9,0.7]) #3rd claster
], dtype=object)




#control or comparing data
#Example:
data_control = np.array([
    [1,1,1.5,0.5,1], #1st claster
    [1.1,1.2,0.8,1.3,1.0, 1.1], #2nd claster
    [1,1.1,1.3,1.05,0.95]#3rd claster
], dtype=object)







# ### Analyzing the data we read from the file
#  The text file must consist of lines (clusters), each of which contains numbers separated by a space.


#To choose this option you should uncomment this part and comment the first
#data_exp = read_file('experiment.txt')#Your path to the file
#data_control = read_file('control.txt')#Your path to the file







# You can also read csv files. For example, if your data is in Excel. You can save them in csv format. And upload it here. In this case, each row will be considered as a separate cluster. Use different data to load different data (experiment and control).


#To choose this option you should uncomment this part and comment the first
#data_exp = read_file_csv(r'C:\Users\Igor\Desktop\example_data.csv')#Your path to the file
#data_control = read_file_csv(r'C:\Users\Igor\Desktop\example_data.csv')#Your path to the file





analyze(data_exp, data_control)




