# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:17:51 2019

@author: nidhi
"""


# -*- coding: utf-8 -*-
"""
This file is for preprocessing channel arrays
"""

import numpy as numpy
import math
import os
import random
import Paths as paths
from scipy import stats

#Loading data from .txt files into array
#Loading 3 channel data samples

traindata =[]
trainlabels=[]
testdata=[]
testlabels=[]

path, dirs, files = next(os.walk(paths.sourceDataFolder))
test_list = []

folderCount = len(dirs)
print(folderCount)
test_list = random.sample(range(1,folderCount) ,int(folderCount - round(0.8*folderCount)))
print(test_list)

def appendDataAndLabels(file,dataArray,labelArray):
        tData = numpy.loadtxt(paths.sourceDataFolder+file+paths.sep+paths.channel1 , encoding= 'unicode_escape', skiprows=7)
        tLabels = numpy.loadtxt(paths.sourceDataFolder+file+paths.sep+paths.labels)
        nLabels = math.floor(len(tData) / paths.samplesPerLabel)
        tData = tData[:-(int(len(tData) -  (paths.samplesPerLabel * nLabels)))]
        tLabels = tLabels[:-(int(len(tLabels) - nLabels))]
        dataArray.append(tData)
        labelArray.append(tLabels)

for file in os.listdir(paths.sourceDataFolder):
    if int(file) not in test_list:
        appendDataAndLabels(file,traindata,trainlabels)
    else:
        appendDataAndLabels(file,testdata,testlabels)
    #data.append(numpy.loadtxt(paths.sourceDataFolder+files+paths.sep+paths.channel1 , skiprows=7))
    
traindata=stats.zscore(numpy.concatenate(traindata,axis=0))
trainlabels = numpy.concatenate(trainlabels,axis=0)
testdata = stats.zscore(numpy.concatenate(testdata,axis=0))
testlabels = numpy.concatenate(testlabels,axis=0)
print(len(traindata))
print(len(trainlabels))
print(len(testdata))
print(len(testlabels))


numpy.save("traindata1dagain.npy",traindata)
numpy.save("trainlabels1dagain.npy",trainlabels)
numpy.save("testdata1dagain.npy",testdata)
numpy.save("testlabels1dagain.npy",testlabels)









 






 
