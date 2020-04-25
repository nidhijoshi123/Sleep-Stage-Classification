# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:12:02 2019

@author: Nidhi
"""

import numpy as numpy
import math
import os
import Paths as paths
from scipy import stats
from keras.models import load_model
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import manifold
import pandas as pd
import keras.backend as K
import GDV


def appendDataAndLabels(file,dataArray,samplesPerLabel,labelArray,channel):
        tData = numpy.loadtxt(paths.sourceDataFolder+file+paths.sep+channel ,encoding='unicode_escape', skiprows=7)
        tLabels = numpy.loadtxt(paths.sourceDataFolder+file+paths.sep+paths.labels)
        nLabels = math.floor(len(tData) / samples)
        tData = tData[:-(len(tData) -  (samples * nLabels))]
        print(len(tData))
        tLabels = tLabels[:-(len(tLabels) - nLabels)]
        dataArray.append(tData)
        labelArray.append(tLabels)
        
def removeResiduals3(Y,X,remIndY):
    count = 0
    for val in remIndY:
        val = val - count
        X = numpy.delete(X,list(range(val*samples,(val*samples+samples))))
        count = count + 1
    Y = list(filter(lambda a: a != -1, Y))
    return [X,Y]

def predict(data,labels,channel):
    data= stats.zscore(numpy.concatenate(data,axis=0))
    labels = numpy.concatenate(labels,axis=0)
    model = load_model(channelModelDic.get(channel))

    remIndY = [i for i, e in enumerate(labels) if e == -1]
    data,labels = removeResiduals3(labels,data,remIndY)

    X = data.reshape(int(len(data) / samples),samples,1)
    Y_pred = model.predict(X)
    return X,Y_pred


        
def plotMDSSeaborn(X,labels,plotName,palette):
    plt.figure()
    mds = manifold.MDS(2, max_iter=300)
    #mds = manifold.TSNE(perplexity=40 , n_components=n_components)
    Y = mds.fit_transform(X.astype(numpy.float64))
    df = pd.DataFrame({"dim1" : Y[:,0], "dim2" : Y[:, 1] , "pos" : labels})
    #palette = dict(zip(set(labels), sns.color_palette(n_colors=16)))
    sns.scatterplot(  x="dim1", y="dim2", hue="pos", data=df,   palette=palette)
    plt.savefig(plotName)

def plotMDS(X,labels,plotname,cm):
    plt.figure()

    mds = manifold.MDS(n_components=2, max_iter=300, n_init=1)
    #mds = manifold.TSNE(perplexity=15 , n_components=n_components)
    Y = mds.fit_transform(X.astype(numpy.float64))

    for i in range(0,5):
       plt.scatter([Y[:, 0][ind] for ind,j in enumerate(labels) if j==i], [Y[:, 1][ind] for ind,j in enumerate(labels) if j==i], c=cm.get(i))
    #plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap=cmap)
    #plt.scatter(Y[:, 0], Y[:, 1])
    plt.title("MDS")
    plt.show()
    plt.savefig(plotname)
    
def plotGDV():
    gdvList = numpy.load('gdvList_patient1_channel1.npy')
    layerList = [1,4,7,10,13,16,19,22,25,28,31,34,37,38]
    plt.figure()
    plt.plot(layerList,gdvList,'ro-')
    plt.xlabel('Layer')
    plt.ylabel('GDV')
    plt.savefig('gdvPlot.png')   
        

#file = '1'
channelModelDic = {paths.channel1:'finalmodel_channel1.h5',paths.channel2:'finalmodel_channel2.h5',paths.channel3:'finalmodel_channel3.h5'}
channelDic={paths.channel1:'channel1',paths.channel2:'channel2',paths.channel3:'channel3'}
channelList=[paths.channel1,paths.channel2,paths.channel3]
#channelList = [paths.channel3]
samples = paths.samplesPerLabel
outputlayerList=[i for i in range(0,37,3)]
outputlayerList.append(37)
cm = {0:'r' , 1:'g' , 2:'b' , 3:'k' , 4:'y'}
filesList = [str(i) for i in range(1,69)]
#filesList = ['01','02','03','04','05','06','07','08','09']


for channel in channelList:
    for file in os.listdir(paths.sourceDataFolder):
        if str(file) in filesList:
            data=[]
            labels=[]
            appendDataAndLabels(file,data,samples,labels,channel)
            X,y = predict(data,labels,channel)
            ylabels = list(numpy.argmax(y,axis=-1))
    
            model = load_model(channelModelDic.get(channel))
            #gdvList=[]


            for lIndex in outputlayerList:
                get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[lIndex].output])
                layer_output = get_layer_output([X])[0]
                if len(layer_output.shape) == 3:
                    layer_output = numpy.reshape(layer_output,(layer_output.shape[0],layer_output.shape[1]*layer_output.shape[2]))
                plotMDS(layer_output,ylabels,'Patient '+str(file)+'_'+channelDic.get(channel)+'_'+'layer '+str(lIndex+1)+'mds.png',cm)
                gdv = GDV.discrimination_value(layer_output,numpy.array(ylabels),reduce=0.2)
                gdvList.append(gdv)
      
            numpy.save('gdvList_'+'patient '+str(file)+'_'+channelDic.get(channel)+'.npy' , gdvList)
    
   