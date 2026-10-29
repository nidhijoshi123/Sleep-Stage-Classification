Data used to feed the networks in this project is strictly confidential to the institute and is hence not a part of the repo.

Please refer to the master thesis report :mt_Nidhi.pdf as a useful prerequisite to the implementation.

The file _Data_Prepocessing_1Channel_ contains all the code for preprocessing like reading data in the form of numpy arrays, concatenating training and test patient's data and labels separately and applying z_score normalization.

_Newapproachwithdropout_ contains all commands for model designing and fitting the data to this model.

Hypnodensity plots are plotted using the file hypnoplot.py. These are the softmax layer outputs (of the model in _Newapproachwithdropout_)for every patient plotted against time in seconds.

mdsplot.py and gdv.py were used for plotting MDS (Multidimensional Scaling) and GDV (Generalized Discrimination Value) respectively.


