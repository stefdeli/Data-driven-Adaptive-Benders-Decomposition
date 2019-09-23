# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:22:26 2018

@author: bvandenb
"""
from gurobipy import *
import gurobipy as gb
#import Data_Load
import numpy as np
import matplotlib.pyplot as plt
#import Lib_Variables
#import Lib_ObjFnct
#import Lib_Constraints
#import Lib_ResExtr
#import Build_constraints
from defaults import SchedulingHorizon,NScen,multicut,nodenb
###################################################################################################
#Clustering Issues 
#from kshape.core import kshape, zscore #import kshape library
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
#from pyclustering.cluster.kmedoids import kmedoids
import kmedoids
from sklearn.metrics.pairwise import pairwise_distances #for k-medoids

#create array with all the data for each scenario to be able to apply the clustering method
def _build_clusters(self,clust_num,method):
    timeseries=[]
    for k in range(len(self.data.windfarms)):
        timeseries.append([])
        for i in range(1,NScen+1):
#            print(k,i)
#            print(self.data.windscen[k+1]['{0}'.format(i)].values)
            timeseries[k].append(self.data.windscen[k+1]['{0}'.format(i)].values)
    time_series=[]
    for i in range(1,NScen+1):
        l=list()
        for k in range(len(self.data.windfarms)):
            l+=timeseries[k][i-1].tolist()
        time_series.append(l)
        
            
            
 #k-shape from kshape.core           
    if method=='k_shape':    
#selection of the number of clusters that should be done
        cluster_num =clust_num
#apply clustering method
        cluster = kshape(zscore(time_series, axis=1), cluster_num)
        self.clusters=[]
        for k in range(len(cluster)):
            self.clusters.append(cluster[k][1])
#kshape from tslearn (recommended by Paparrizos)
#        from tslearn.clustering import KShape
#        from tslearn.utils import to_time_series_dataset
#        formatted_dataset = to_time_series_dataset(time_series)
#        ks=ks=KShape(n_clusters=cluster_num, verbose=False)
#        y_pred=ks.fit_predict(formatted_dataset)
#        self.clusters=[]
#        for n in range(cluster_num):
#            self.clusters.append([])
#        for k in range(NScen):
#            self.clusters[y_pred[k]].append(k)
        

    if method=='k_means':
#k-means clustering 
        n_clusters=clust_num
        kmeans = KMeans(n_clusters, random_state=0).fit(time_series)
        kmeans.labels_
        self.clusters=[]
        for n in range(n_clusters):
            self.clusters.append([])
        k=0
        for i in range(NScen):
            k+=1
            self.clusters[kmeans.labels_[i]].append(k-1)
        
    if method=='hierar':
#hierarchical clustering
        n_clusters=clust_num
        cluster = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward').fit_predict(time_series)  
        self.clusters=[]
        for n in range(n_clusters):
            self.clusters.append([])
        k=0
        for i in range(NScen):
            k+=1
            self.clusters[cluster[i]].append(k-1) 
    
    if method=='k_medoids':
#k-medoids clustering
        n_clusters=clust_num
        D = pairwise_distances(np.array(time_series), metric='euclidean')
        M, C = kmedoids.kMedoids(D, n_clusters) #M is a list of medoids and C a list of cluster (repartition of the scenarios in the clusters, only number (ID) of scenario not data)
        self.clusters=[]
        self.medoids=list(M)
        for i in range(n_clusters):
            self.clusters.append(list(C[i]))
            
##proba of each cluster           
#    self.data.scenprob_clust=[]        
#    for c in range(len(self.clusters)):
#        p=sum(self.data.scenprob[s+1] for s in self.clusters[c])
#        self.data.scenprob_clust.append(p)
#        
    
##############################Plot cluster attribution###############
##multiplot
#plt.figure(num=5,figsize=(25, 20))
#nb=np.arange(len(time_series[0]))
#for k in range(len(time_series)):
#    plt.subplot(5,5,k+1)
#    plt.plot(nb,time_series[k],label='{0}'.format(k))
#plt.xlabel('time')
#plt.ylabel('Wind production forcast')
#plt.title('Scenarios')
#plt.legend()
#plt.show
#
##single plot        
#plt.figure(num=6,figsize=(25, 20))
#nb=np.arange(len(time_series[0]))
#for k in range(len(time_series)):
#    plt.plot(nb,time_series[k],label='{0}'.format(k))
#plt.xlabel('time')
#plt.ylabel('Wind production forcast')
#plt.title('Scenarios')
#plt.legend()
#plt.show  
#
##color per group   
#color_set=['xkcd:black','xkcd:blue','xkcd:green','xkcd:red','xkcd:pink','xkcd:sienna','xkcd:purple','xkcd:orange','xkcd:gold','xkcd:teal']   
#plt.figure(num=6,figsize=(25, 20))
#nb=np.arange(len(time_series[0]))
#for k in range(len(time_series)):
#    plt.subplot(5,5,k+1)
#    for i in range(len(clusters)):
#        if k in clusters[i]:
#            col=color_set[i]
#    plt.plot(nb,time_series[k],col,label='{0}'.format(k))
#plt.xlabel('time')
#plt.ylabel('Wind production forcast')
#plt.title('Scenarios')
#plt.legend()
#plt.show
