# Introduction and Parameters of the Code
Biological experiments often include data with intrinsic hierachy. For example, a number of experiments conducted on different days or several studies performed using different samples (such as mice). These are so called clusters that contain experiments of a lower level of hierachy. The experimental conditions on different days are not always the same. The cells from one mouse differ less than from those from the diffrent mice. An experimenter should consider these features of data in order to correctly evaluate differences between compared groups. Erroneous assumptions can lead to pseudo-replication and false positive results. This, in turn, complicates correct assessment of statistical power and impairs optimal planning of experiments. 

We designed a simple data simulator that helps an experimenter quickly generate data and learn how the outcomes of data analysis depend on the construction of the experiment, on the intrinsic data features and on the ways in which the data are processed. There are several parameters in this code:
* **intrinsic data features** 
	* true values
	* the variance of inter-cluster data (cluster-to-cluster variance)
	* the variance of intra-cluster data (experiment-to-experiment variance inside a cluster)
* **features of experiment consrtuction**
	* The number of clusters 
	* The number of per cluster experiments 
* **the choice of data processing type**
	* p-values calculation based on pooled data (disregarding the hierachy of the data)
	* p-values calculation based on clusters 
	* adjusted p-values calculation 
		
For more information read [Methods](https://github.com/juliaLopanskaia/biostastics_article/blob/master/docs/methods.md)
