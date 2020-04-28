Sorry that I am unable to share my data with you since it is confidential.

Please read the master thesis report :mt_Nidhi.pdf, before jumping to code section.

The file Data_Precpocessing_1Channel contains all the code for preprocessing like reading data in the form of numpy arrays, concatenating training and test patient's data and labels separately and applying z_score normalization.

Newapproachwithdropout contains all commands for model designing and fitting the data to this model.

Hypnodensity plots are plotted using the file hypnoplot.py. These are nothing but the softmax layer outputs (of the model in Newapproachwithdropout)for every patient plotted against time in seconds.

mdsplot.py and gdv.py were used for plotting mds and gdv respectively.

plotlabels is not very important. You can ignore that.


