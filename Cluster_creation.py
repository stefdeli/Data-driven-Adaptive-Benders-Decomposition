# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:20:14 2019

For GNU GLP v3:

{{Data-driven Adaptive Benders Decomposition for the Stochastic Unit Commitment Problem}}
Copyright (C) {{ 2019 }}  {{ ETH Zurich }}

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

@author: bvandenb
"""

#    from gurobipy import *
#import argparse
import pandas as pd
#import gurobipy as gb
#import matplotlib.pyplot as plt
import numpy as np
#from kshape.core import kshape, zscore #import kshape library
#from sklearn.cluster import KMeans
#from sklearn.cluster import AgglomerativeClustering
#import Data_Load
#import Lib_Variables_multiprocess_Parallel
#import Lib_ObjFnct_cluster_multiprocess_Parallel
#import Lib_Constraints_multiprocess_Parallel
#import Lib_ResExtr
#import Lib_Cluster
from time import time, sleep
#import Build_constraints
from defaults import SchedulingHorizon,NScen,multicut,nodenb,windfarms_file,WindScen_file_1,WindScen_file_2,WindScen_file_3,WindScen_file_4,WindScen_file_5,WindScen_file_6,WindScen_file_7,WindScen_file_8,WindScen_file_9,WindScen_file_10,WindScen_file_11,WindScen_file_12,WindScen_file_13,WindScen_file_14,WindScen_file_15
import kmedoids
from sklearn.metrics.pairwise import pairwise_distances #for k-medoids
####for PARALLELIZATION
#do not forget to create the engines in the Anaconda prompt
#ipcluster start -n 4 :for 4 engines (activate sp27n to activate the good environment)
#import ipyparallel as ipp
#rc = ipp.Client()
#dview = rc[:] # use all engines
import multiprocessing as mp
#from multiprocessing import Pool
#from pathos.multiprocessing import ProcessingPool as Pool
from joblib import Parallel, delayed


def clust_creation(x):    
#    T0=time()
    windscen={}
    windscen[1]=pd.read_csv(WindScen_file_1,index_col=0)
    windscen[2]=pd.read_csv(WindScen_file_2,index_col=0)
    windscen[3]=pd.read_csv(WindScen_file_3,index_col=0)
    windscen[4]=pd.read_csv(WindScen_file_4,index_col=0)
    windscen[5]=pd.read_csv(WindScen_file_5,index_col=0)
    windscen[6]=pd.read_csv(WindScen_file_6,index_col=0)
    windscen[7]=pd.read_csv(WindScen_file_7,index_col=0)
    windscen[8]=pd.read_csv(WindScen_file_8,index_col=0)
    windscen[9]=pd.read_csv(WindScen_file_9,index_col=0)
    windscen[10]=pd.read_csv(WindScen_file_10,index_col=0)
    windscen[11]=pd.read_csv(WindScen_file_11,index_col=0)
    windscen[12]=pd.read_csv(WindScen_file_12,index_col=0)
    windscen[13]=pd.read_csv(WindScen_file_13,index_col=0)
    windscen[14]=pd.read_csv(WindScen_file_14,index_col=0)
    windscen[15]=pd.read_csv(WindScen_file_15,index_col=0)
    windinfo = pd.read_csv(windfarms_file, index_col = 0)         
    windfarms =windinfo.index.tolist()
    scenprob_init = {s: 1.0/NScen for s in range(1,NScen+1)}
    timeseries=[]
    for k in range(len(windfarms)):
        timeseries.append([])
        for i in range(1,NScen+1):
            timeseries[k].append(windscen[k+1]['{0}'.format(i)].values)
    time_series=[]
    for i in range(1,NScen+1):
        l=list()
        for k in range(len(windfarms)):
            l+=timeseries[k][i-1].tolist()
        time_series.append(l)
    #use k-medoid to generate the clusters and the centroids (significant scenario of every cluster)
    n_clusters=x
    D = pairwise_distances(np.array(time_series), metric='euclidean')
    M, C = kmedoids.kMedoids(D, n_clusters) #M is a list of medoids and C a list of cluster (repartition of the scenarios in the clusters, only number (ID) of scenario not data)
    cluster=[]
    for i in range(n_clusters):
        cluster.append(list(C[i]))
    medoid_prob={p:sum(scenprob_init[i+1] for i in cluster[p]) for p in range(len(cluster))}
    scenprob={}
    clusters=[]
    for i in range(len(cluster)):
        add_med=list(M)
        add_med.pop(i)
        clusters.append(cluster[i]+add_med)
        scenprob[i]={}
        for j in clusters[i]:
            if (j in cluster[i]):
                scenprob[i][j+1]=scenprob_init[j+1]
            else:
                scenprob[i][j+1]=medoid_prob[list(M).index(j)]
    return(clusters,scenprob)            
        
H=clust_creation(30)
print(H)
