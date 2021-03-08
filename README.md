# data_simulator.py
This is a simple simulator of experimental data. Use it in case one wants to understand how false positives and false negatives can occur (explore how p-value of your result is effected by the number of experiments, the number of days or samples for your study).

![GitHub Logo](/images/picture.png)
Format: ![Alt Text](url)



Biological experiments often include data with intrinsic hierachy. For example, a number of experiments conducted in different days or several studies performed using different samples (such as mice). These are so called clusters that contain experiments of a lower level of hierachy. The experimental conditions in different days are not always the same. The cells from one mice differ less than from those from the other mice. An experimentator should consider these features of data because the expoit of p-values calculation requires genuinely independent data.

An experimentator usually question. How to correctly set an experiment? Should it be ten days with one experiment or ten accurate experiments in one day? How should I do data processing?
We designed a simple data simulator that helps an experimentator quickly generate data and learn how the results of the experiment depend on the construction of the experiment, data intrinsic features and even data processing. There are several parameters in this code:
		(intrinsic data features) 1) true values
					   2) the variance of inter-cluster data (cluster-to-cluster variance)
					   3) the variance of intra-cluster data 
		(features of experiment consrtuction) 1) The number of clusters 
					2) The number of per day experiments 
		(the choice of data processing type) 1) p-values calculation based on pooled data (ignorence of data hierachy зкщзукешуы)
		2) p-values calculation based on clusters 
		3) correct p-values calculation 
		
For more information on methods and tips on how to correctly set experiment and process the data read 


INSTALLATION
python -m pytest tests
https://docs.pytest.org/en/latest/usage.html#calling-pytest-through-python-m-pytest

Можно установить проект в качестве пакета (в режиме разработчика) с помощью pip install -e <проект>, чтобы добавить корень проекта в sys.path. 
https://stackoverflow.com/questions/20971619/ensuring-py-test-includes-the-application-directory-in-sys-path
