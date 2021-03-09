# Introduction and Parameters of the Code
Biological experiments often include data with intrinsic hierachy. For example, a number of experiments conducted in different days or several studies performed using different samples (such as mice). These are so called clusters that contain experiments of a lower level of hierachy. The experimental conditions in different days are not always the same. The cells from one mice differ less than from those from the other mice. An experimentator should consider these features of data because the expoit of p-values calculation requires genuinely independent data. Errors in these means can lead to pseudo-replication and false positive results. This, in turn, complicates correct assessment of statistical power and impairs optimal planning of experiments. 

We designed a simple data simulator that helps an experimentator quickly generate data and learn how the results of the experiment depend on the construction of the experiment, data intrinsic features and even data processing. There are several parameters in this code:
* (intrinsic data features) 
	1. true values
	2. the variance of inter-cluster data (cluster-to-cluster variance)
	3. the variance of intra-cluster data (experiment-to-experiment variance inside a cluster)
* (features of experiment consrtuction) 
	* 1. The number of clusters 
	* 2. The number of per cluster experiments 
* (the choice of data processing type) 
	* p-values calculation based on pooled data (ignorence of data hierachy)
	* p-values calculation based on clusters 
	* adjusted p-values calculation 
		
For more information read [Methods](https://github.com/juliaLopanskaia/biostastics_article/blob/master/docs/methods.md)
