# nested_data_simulator.py
This is a simple nested data simulator. It generates two-level experimental data and processes it. The user can test various scenarios, appreciate the importance of using correct multi-level analysis and the danger of neglecting the information about the data structure. Please, feel free to use it in case you aim to vizualize and understand how false positives and false negatives can occur. This simple tool also helps an experimenter correctly set up an optimal experiment in order to achieve required statistical power. 

| ![picture_readme.png](/images/picture_readme.png) | 
|:--:| 
| *Imagine that the 'true' values of 'control' and 'experiment' are equal. The example case on the left has a weak intra-cluster correlation (ICC) and the example case on the right has a strong ICC. Ignoring data clustering in the right case leads to a false positive result. However, proper processing of data could save the day (see p(per-cluster) or p(adjusted)).* |

<br/><br/>
Using this tool we have highlighted some commonly arising mistakes with data analysis and proposed a workflow, in which our simulator could be employed to correctly compare two groups of nested experimental data and to optimally plan new experiments in order to increase statistical power if it is necessary. See our article 'Avoiding common problems with statistical analysis of biological experiments using a simple nested data simulator'.

## Documentation
[Introduction & Parameters](https://github.com/juliaLopanskaia/nested_data_simulator/blob/master/docs/intro_parameters.md)
<br/>
[Methods](https://github.com/juliaLopanskaia/nested_data_simulator/blob/master/docs/methods.md)
<br/>
[Tutorial](https://github.com/juliaLopanskaia/nested_data_simulator/blob/master/docs/tutorial.md)
