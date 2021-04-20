# Introduction and Parameters of the Code
Biological experiments often include data with intrinsic hierachy. For example, a number of experiments conducted on different days or several studies performed using different samples (such as mice). These are so called clusters that contain experiments of a lower level of hierachy. The experimental conditions on different days are not always the same. The cells from one mouse differ less than from those from the diffrent mice. An experimenter should consider these features of data in order to correctly evaluate differences between compared groups. Erroneous assumptions can lead to pseudo-replication and false positive results. This, in turn, complicates correct assessment of statistical power and impairs optimal planning of experiments.

Here we present a tool that can generate two-level experimental data and processes it. It also can analyze input data to test whether the result is reliable. Both processes rely on data parameters that we state here and on methods described in the next chapter ([Methods](https://github.com/juliaLopanskaia/nested_data_simulator/blob/master/docs/methods.md)).

For the task of data generation the user should explicitly enter the parameters to the program. This data simulator helps an experimenter quickly generate data and learn how the outcomes of data analysis depend on the intrinsic data features (1 block of parameters), on the construction of the experiment (2 block of parameters) and on the ways in which the data are processed (3 block of parameters).

There are several parameters in this code:
* **intrinsic data features**
	* true values of measurements
	* the variance of inter-cluster data (cluster-to-cluster variance)
	* the variance of intra-cluster data (experiment-to-experiment variance inside a cluster)
* **features of experiment consrtuction**
	* The number of clusters
	* The number of per cluster experiments
* **the choice of data processing type**
	* p-values calculation based on pooled data (disregarding the hierachy of the data)
	* p-values calculation based on clusters
	* adjusted p-values calculation

 If the user wants to simply analyze one's data there is no need to know the parameters beforehand (it is automatically calculated). Due to the fact that in our article we proved that in any scenario an experimenter should use adjusted t-test, the analyzer use this processing method and does not ask a user about it. However it easily can be changed if one wants to and knows python.

For more information read [Methods](https://github.com/juliaLopanskaia/nested_data_simulator/blob/master/docs/methods.md)
