# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:50:19 2019

@author: joshini
"""

import matplotlib.pyplot as plt
import numpy as np

#This method scales the Y values, i.e ,repeats each value 30 times to cover 30s interval  
def scaleWithTime(y):
    all_y=[]
    for i in range(len(y)):
        all_y+= [y[i]] * 30
    return all_y
       
    
#Older approach which led to overlap and white spaces in the plot
def plotAllClassesInOneOldApproach(y_indices,y_0,y_1,y_2,y_3,y_4,filename):
    plt.xlabel('Time in seconds')
    plt.ylabel('P(Y|X)')
    plt.plot(y_indices, scaleWithTime(y_0), 'y',label='C1')
    plt.plot(y_indices, scaleWithTime([sum(x) for x in zip(y_0,y_1)]), 'r',label='C2')
    plt.plot(y_indices, scaleWithTime([sum(x) for x in zip(y_0,y_1,y_2)]), 'g',label='C3')
    plt.plot(y_indices, scaleWithTime([sum(x) for x in zip(y_0,y_1,y_2,y_3)]), 'b',label='C4')
    plt.plot(y_indices, scaleWithTime([sum(x) for x in zip(y_0,y_1,y_2,y_3,y_4)]), 'k',label='C5') 
    plt.legend()
    fig1 = plt.gcf()
    fig1.savefig(filename)
    plt.close()

# =============================================================================
# This method plots classes on top of one another per time interval in the order:  
#     WAKE,REM,N1,N2,N3
# =============================================================================
    
def plotAllClassesInOne(y_indices,y_0,y_1,y_2,y_3,y_4,filename):

    y_0 = np.array(scaleWithTime(y_0))
    y_1 = np.array(scaleWithTime(y_1))
    y_2 = np.array(scaleWithTime(y_2))
    y_3 = np.array(scaleWithTime(y_3))
    y_4 = np.array(scaleWithTime(y_4))
    length = y_0.shape[0]
    #length = len(y_0)
    print(length)
    #length = len(y_0)
    xaxis = np.arange(0, length)
    #xaxis = y_indices
    c1_base = np.zeros(length)
    c1_height = y_0

    c2_base = c1_height
    c2_height = c2_base + y_1

    c3_base = c2_height
    c3_height = c3_base + y_2

    c4_base = c3_height
    c4_height = c4_base + y_3

    c5_base = c4_height
    c5_height = c5_base + y_4


    plt.xlabel('Time in seconds')
    plt.ylabel('P(Y|X)')
    plt.fill_between(xaxis, c1_base, c1_height, label='WAKE', color='y')
    plt.fill_between(xaxis, c2_base, c2_height, label='REM', color='r')
    plt.fill_between(xaxis, c3_base, c3_height, label='N1', color='g')
    plt.fill_between(xaxis, c4_base, c4_height, label='N2', color='b')
    plt.fill_between(xaxis, c5_base, c5_height, label='N3', color='k')
    plt.legend()
    fig1 = plt.gcf()
    fig1.savefig(filename)
    plt.close()
    
    

#Load the numpy arrays for all class labels along with index array for the x axis
y_indices = np.load('yIndices.npy')
y_0 = np.load('Class0Probabilities.npy')
y_1 = np.load('Class1Probabilities.npy')
y_2 = np.load('Class2Probabilities.npy')
y_3 = np.load('Class3Probabilities.npy')
y_4 = np.load('Class4Probabilities.npy')
print('Length of prediction indices: ' + str(len(y_indices)) )
print('****************************************')

#Plot the hypno-densities
plotAllClassesInOne(y_indices,y_0,y_1,y_2,y_3,y_4,'hypno_plot.png')