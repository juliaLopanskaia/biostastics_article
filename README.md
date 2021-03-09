# experiment_simulator.py
This is a simple experiment simulator (it generates experimental data and processes it). The user can test various scenarios, appreciate the importance of using correct multi-level analysis and the danger of neglecting the information about the data structure. Use it in case one wants to understand how false positives and false negatives can occur. This simple tool helps an experimentator correctly set an experiment to achieve strong results. 

^^^
![GitHub Logo](/images/picture.png)

*Imagine that the true values of control and experiment are equal. The experiment on the left has weak intra-cluster correlation and the experiment on the right has strong ICC. Ignoring data clustering in the right case leads to a false positive result. However proper processing of data could save the day (see P(per-cluster) or P(adjusted)).*
^^^

Using this tool we have highlighted some commonly arising mistakes with data analysis and proposed a workflow, in which our simulator could be employed to correctly compare two nested groups of experimental data and to optimally plan new experiments in order to increase statistical power if it is necessary. See our article ' Avoiding common problems with statistical analysis of biological experiments using a simple nested data simulator' (not published yet).

## Documentation
[Methods](https://github.com/juliaLopanskaia/biostastics_article/blob/master/docs/methods.md)

[Tutorial](https://github.com/juliaLopanskaia/biostastics_article/blob/master/docs/tutorial.md)
