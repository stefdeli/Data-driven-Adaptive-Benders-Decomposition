# -*- coding: utf-8 -*-
"""

#"""
#from gurobipy import *
#import gurobipy as gb
#import matplotlib.pyplot as plt
#import numpy as np
#import Data_Load
#import Lib_Variables
#import Lib_ObjFnct_cluster
#import Lib_Constraints
#import Lib_ResExtr
#import Lib_Cluster
##import Build_constraints
#from defaults import SchedulingHorizon,NScen,multicut,nodenb

# Class which can have attributes set
#IMPORT ALL THE PACKAGES
#    from gurobipy import *
import argparse
import pandas as pd
import gurobipy as gb
import matplotlib.pyplot as plt
import numpy as np
#from kshape.core import kshape, zscore #import kshape library
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import Data_Load
import Lib_Variables_multiprocess_Parallel
import Lib_ObjFnct_cluster_multiprocess_Parallel
import Lib_Constraints_multiprocess_Parallel
#import Lib_ResExtr
import Lib_Cluster
from time import time, sleep
#import Build_constraints
from defaults import SchedulingHorizon,NScen,multicut,nodenb,windfarms_file,WindScen_file_1,WindScen_file_2,WindScen_file_3,WindScen_file_4,WindScen_file_5,WindScen_file_6
import kmedoids
from sklearn.metrics.pairwise import pairwise_distances #for k-medoids
import multiprocessing as mp
from joblib import Parallel, delayed

 
#Create the function to extract the function defined in the sub-problem class to 
#run the computation in parallel 
#t is the set of first stage decisions from MP  
def unwrap_self_optmz_sp(t):   
    ss=t[0]
    Resns=t[1]
    Resps=t[2]
    WindDAs=t[3]
    lineflowDAs=t[4]
    t0=time() #strating time to store computational time
#    print(ss,Resns,Resps,WindDAs,lineflowDAs)
    sp=SubProblem(ss,Resns,Resps,WindDAs,lineflowDAs) #create the SP based on the SP class
    sp.model.optimize() #optimize SP
	sens_Resp={}# create dict to store the second stage data for the SP
    sens_Resn={}
    sens_WindDA={}
    sens_lineflowDA={}
    for g in sp.data.generators: #extract the duals from the SP
        sens_Resp[g]={
        t: sp.subproblem_fixedp[g][t].pi for t in range(SchedulingHorizon)}
        sens_Resn[g]={
        t: sp.subproblem_fixedn[g][t].pi for t in range(SchedulingHorizon)}
    for j in sp.data.windfarms:
        sens_WindDA[j]={
        t: sp.subproblem_fixedw[j][t].pi for t in range(SchedulingHorizon)}
    for l in sp.data.linesindex:
        sens_lineflowDA[l]={
        t:sp.subproblem_flow[l][t].pi for t in range(SchedulingHorizon)}
    z_sub=sp.model.Objval
    t1=time()
    H=[ss,(t1-t0),z_sub,sens_Resp,sens_Resn, sens_WindDA,sens_lineflowDA] #create the output
#    print(H)
    return H

#define a function to run the SPs in parallel
def run_sp_parallel(scenar,Resn,Resp,WindDA,lineflowDA,jobs):
#    if __name__ == '__main__':
    NScen=len(scenar)
	#parallelized way
    results=Parallel(n_jobs=jobs)(delayed(unwrap_self_optmz_sp)(t) for t in list(zip(scenar,[Resn]*NScen,[Resp]*NScen,[WindDA]*NScen,[lineflowDA]*NScen))) # command with joblib to run in parallel, delayed is to wait and run in the same time 
#    #serial way
#    results=[]
#    for s in list(zip(scenar,[Resn]*NScen,[Resp]*NScen,[WindDA]*NScen,[lineflowDA]*NScen)):
#        results.append(unwrap_self_optmz_sp(s))
    for r in results: # aggrgate the results of the SPs to create the input for MP and management strategies (cluster creation...)
        out=r
        sc=out[0]
        sens_Resp[sc]=out[3]
        sens_Resn[sc]=out[4]
        sens_WindDA[sc]=out[5]
        sens_lineflowDA[sc]=out[6]
        z_sub[sc]=out[2]
    return(sens_Resp,sens_Resn,sens_WindDA,sens_lineflowDA,z_sub)
            
class expando(object):
    pass
############################################################################################################################
#class for subproblem
#define a class to be able to built a standard object for the sub-problems
class SubProblem:
    def __init__(self,scenar,Resn,Resp,WindDA,lineflowDA): #initialisation of the SP        
        self.data = expando()        
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._load_data()
#        self.data.scenario=scenario
        self._build_model(scenar,Resn,Resp,WindDA,lineflowDA)
#        self.update_fixed_vars(MP)
    
    def _load_data(self):
        Data_Load._load_network(self)     
        Data_Load._load_generator_data(self)
        Data_Load._load_wind_data(self)
        Data_Load._load_intial_data(self)
       
                   
    def optimize(self):
        t0=time()
        self.model.optimize()
        t1=time()
        print(t1-t0)       
                   
        
    def _build_model(self,w,Resn,Resp,WindDA,lineflowDA):
        self.model = gb.Model()        
        self.model.setParam('OutputFlag',0)
        Lib_Variables_multiprocess_Parallel.build_variables_sb(self,w)
        Lib_Constraints_multiprocess_Parallel.build_subproblem_constr(self,w,Resn,Resp,WindDA,lineflowDA)
        Lib_ObjFnct_cluster_multiprocess_Parallel.build_objective_subpb(self,w)         
        self.model.update()
        
    def update_fixed_vars(self,MP):
        Lib_Constraints_multiprocess.update_fixed_vars(self,MP)
        self.model.update()
    
    def _store_data(self,MP):
        for j in self.data.demand.index.tolist():
            MP.data.store_lshed[j]={}
            for t in range(SchedulingHorizon):
                MP.data.store_lshed[j][t]={}
                for n in range(1,NScen+1):
                    MP.data.store_lshed[j][t][n]=self.variables.lshed[j][t][n].x
        for j in self.MP.data.windfarms:
            MP.data.store_wspill[j]={}
            for t in range(SchedulingHorizon):
                MP.data.store_wspill[j][t]={}
                for n in range(1,NScen+1):
                    MP.data.store_wspill[j][t][n]=self.variables.wspill[j][t][n].x
#################################################################################################################
#class for master problem
#define a class to be able to built a standard object for the master-problems
class MasterProblem:
    def __init__(self,clust_num,method,scenar,proba,epsilon=1e-6,delta=1e-6,IterLimit=2000,Exit=0):        
        self.data = expando()        
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._init_benders_params(epsilon,delta,IterLimit,Exit,clust_num,scenar,proba,method)
        self._load_data()            
        self._build_model()
#        Lib_Cluster._build_clusters(self,clust_num,method) #when clusters are fixed at the beginning based on weather forecasts 

    def optmz_sp(t): #to optimize SP in a loop 
        ss=t[0]
        Resns=t[1]
        Resps=t[2]
        WindDAs=t[3]
        lineflowDAs=t[4]
        t0=time()
        sp=SubProblem(ss,Resns,Resps,WindDAs,lineflowDAs)
        sp.model.optimize()
        sens_Resp={}
        sens_Resn={}
        sens_WindDA={}
        sens_lineflowDA={}
        for g in sp.data.generators:
            sens_Resp[g]={
            t: sp.subproblem_fixedp[g][t].pi for t in range(SchedulingHorizon)}
            sens_Resn[g]={
            t: sp.subproblem_fixedn[g][t].pi for t in range(SchedulingHorizon)}
        for j in sp.data.windfarms:
            sens_WindDA[j]={
            t: sp.subproblem_fixedw[j][t].pi for t in range(SchedulingHorizon)}
        for l in sp.data.linesindex:
            sens_lineflowDA[l]={
            t:sp.subproblem_flow[l][t].pi for t in range(SchedulingHorizon)}
        z_sub=sp.model.Objval
        t1=time()
#        H=[ss,(t1-t0),z_sub,sens_Resp,sens_Resn, sens_WindDA,sens_lineflowDA]
#        print(H)
        return [ss,(t1-t0),z_sub,sens_Resp,sens_Resn, sens_WindDA,sens_lineflowDA]
#        return ss
#        sleep(10)
#        return sp.model.Objval
    
        
    def optimize(self,clust_num):
        self.model.optimize()     
#        self.model.write('firststage.mst')
        self.runtime_mp[self.cut]=self.model.runtime
        self.nzlist_mp[self.cut]=self.model.NumNZs
        self.cst_mp[self.cut]=self.model.NumConstrs
#build submodels         
        for i in self.data.generators:
            self.Resp_Fx[i]={}
            self.Resn_Fx[i]={}
            for t in range(SchedulingHorizon):
#                print(self.variables.Resp[i][t])
                self.Resp_Fx[i][t]=self.variables.Resp[i][t].x 
                self.Resn_Fx[i][t]=self.variables.Resn[i][t].x
        for j in self.data.windfarms:
            self.WindDA_Fx[j]={}
            for t in range(SchedulingHorizon):   
                self.WindDA_Fx[j][t]=self.variables.WindDA[j][t].x
        for l in self.data.linesindex:
            self.lineflowDA_Fx[l]={}
            for t in range(SchedulingHorizon):
                self.lineflowDA_Fx[l][t]=self.variables.lineflowDA[l][t].x
#        Resn=self.Resn_Fx
#        Resp=self.Resp_Fx
#        WindDA=self.WindDA_Fx
#        lineflowDA=self.lineflowDA_Fx
#Rescale these values to avoid complications 
        for t in range(SchedulingHorizon):
            for i in self.data.generators:
                if abs(self.Resp_Fx[i][t])<=1e-8:
                    self.Resp_Fx[i][t]=0
                if abs(self.Resn_Fx[i][t])<=1e-8:
                    self.Resn_Fx[i][t]=0
            for j in self.data.windfarms:
                if abs(self.WindDA_Fx[j][t])<=1e-8:
                    self.WindDA_Fx[j][t]=0
            for l in self.data.linesindex:
                if abs(self.lineflowDA_Fx[l][t])<=1e-8:
                    self.lineflowDA_Fx[l][t]=0
        return [self.Resp_Fx,self.Resn_Fx,self.WindDA_Fx,self.lineflowDA_Fx,self.model.Objval]
#        self.submodels={s:SubProblem(self,s) for s in range(1,NScen+1)}
#        print(self.submodels)
#        print("build submodel*****************************************************************************")
#update fixed variables for submodels
#        #for python 2.7
#        [sm.update_fixed_vars(self) for sm in self.submodels.itervalues()]
#        #for python 3.7
#        [sm.update_fixed_vars(self) for sm in iter(self.submodels.values())]
#        print(self.submodels)
#        for i in self.data.generators:
#            self.Resp_Fx[i]={}
#            self.Resn_Fx[i]={}
#            for t in range(SchedulingHorizon):
#                self.Resp_Fx[i][t]=self.variables.Resp[i][t].x 
#                self.Resn_Fx[i][t]=self.variables.Resn[i][t].x
#        for j in self.data.windfarms:
#            self.WindDA_Fx[j]={}
#            for t in range(SchedulingHorizon):   
#                self.WindDA_Fx[j][t]=self.variables.WindDA[j][t].x
#        for l in self.data.linesindex:
#            self.lineflowDA_Fx[l]={}
#            for t in range(SchedulingHorizon):
#                self.lineflowDA_Fx[l][t]=self.variables.lineflowDA[l][t].x
##Rescale these values to avoid complications 
#        for t in range(SchedulingHorizon):
#            for i in self.data.generators:
#                if self.Resp_Fx[i][t]<=1e-8:
#                    self.Resp_Fx[i][t]=0
#                if self.Resn_Fx[i][t]<=1e-8:
#                    self.Resn_Fx[i][t]=0
#            for j in self.data.windfarms:
#                if self.WindDA_Fx[j][t]<=1e-8:
#                    self.WindDA_Fx[j][t]=0
#            for l in self.data.linesindex:
#                if abs(self.lineflowDA_Fx[l][t])<=1e-8:
#                    self.lineflowDA_Fx[l][t]=0
#########################################################################################################################        
##PARALLELIZATION
##        dview.scatter('subpb',self.submodels.values())
##        def f(x):
##            x.model.optimize()
##        results = []
##        results = 
##        from joblib.externals.loky import set_loky_pickler
##        set_loky_pickler()
##        set_loky_pickler('pickle')
##, backend="threading"
#        Parallel(n_jobs=1, backend="threading")\
#            (delayed(unwrap_self_optmz_sp)(i) for i in zip([self]*len(self.submodels.values()), self.submodels.values()))
#            
##        print(results)
#
#        
##        p=Pool(processes=2)
##        print(self.submodels.values())
##        list_sp=self.submodels.values()
##        p.map(unwrap_self_optmz_sp,zip([self]*len(list_sp),list_sp))
##        list_sp=self.submodels.values()
##        p.map(self.optmz_sp,list_sp)
#            
##        y= dview.apply(f,self.submodels.values())
##        y = dview.gather('y').get()
##        print(y)

#        if __name__ == '__main__':
#            print('depart')
#            p=mp.Pool(20)
#            print('start')
##               print(list(zip(range(1,NScen+1),[Resn]*NScen,[Resp]*NScen,[WindDA]*NScen,[lineflowDA]*NScen)))
#            results=[p.map_async(unwrap_self_optmz_sp,t) for t in list(zip(range(1,NScen+1),[Resn]*NScen,[Resp]*NScen,[WindDA]*NScen,[lineflowDA]*NScen))]
##            p.close()
##            p.join()
#            print(results)
#            print('toc')
##                for r in results:
##                    print(r)
##                    out=r.get()
##                    print(out)
##                    sc=out[0]
##                    sens_Resp[sc]=out[3]
##                    sens_Resn[sc]=out[4]
##                    sens_WindDA[sc]=out[5]
##                    sens_lineflowDA[sc]=out[6]
##                    z_sub[sc]=out[2]
##                    print(out[1])
#            print(sens_Resp,sens_Resn,sens_WindDA,sens_lineflowDA,z_sub)
#            print('end')
        
#        print(list(zip(range(1,NScen+1),[Resn]*NScen,[Resp]*NScen,[WindDA]*NScen,[lineflowDA]*NScen)))
#        print(zip(range(1,NScen+1),[Resn]*NScen,[Resp]*NScen,[WindDA]*NScen,[lineflowDA]*NScen))
#        for t in zip(range(1,NScen+1),[Resn]*NScen,[Resp]*NScen,[WindDA]*NScen,[lineflowDA]*NScen):
#            J=unwrap_self_optmz_sp(t)
#            print(J)
                    
#        run_sp_parallel(NScen,Resn,Resp,WindDA,lineflowDA)
##########################################################################################################################        
##SERIALIZATION        
#        #for python 2.7
#        #[sm.optimize() for sm in self.submodels.itervalues()]
#        #for python 3.7
#        [sm.optimize() for sm in iter(self.submodels.values())]
##        print("opt sb**************************************************************************************")
#########################################################################################################################
#update bounds based on submodel rebuild
#        self._update_bounds()
#        self._save_vars()
#        print("save vars**********************************************************************************")
#build cuts till absolute and relative tolerance are reached
#        while self.Exit==0:
#            print("Benders step : {0}*************************************************************".format(len(self.data.cutlist)))
#            print("Upper bound:{0}".format(self.data.ub))
#            print("Lower bound :{}".format(self.data.lb))


#Change clust num manually of sytematically
    def _change_clust_num(self): 
##Manual change of cluster nb 
##            if self.cut==3:
##                Lib_Cluster._build_clusters(self,3,method)
##                print(self.clusters)
#            if self.cut==10:
#                self.clust_num=15
##                Lib_Cluster._build_clusters(self,5,method)
##                print(self.clusters)
#            if self.cut==20:
#                self.clust_num=17
##                Lib_Cluster._build_clusters(self,10,method)
##                print(self.clusters)
##            if self.cut==16:
##                Lib_Cluster._build_clusters(self,15,method)
##                print(self.clusters)
##            if self.cut==19:
##                Lib_Cluster._build_clusters(self,18,method)
##                print(self.clusters)
#            if self.cut==30:
#                self.clust_num=20
##                Lib_Cluster._build_clusters(self,20,method)
##                print(self.clusters)
##            self._do_benders_step()
#More Systematic Way
            #print(self.data.lb)
            #print(self.lb[self.cut])
            #print(abs(self.data.lb-self.lb[self.cut]))
            if (abs(self.data.lb-self.lb[self.cut])<1867) and (self.clust_num<len(self.scenar)-5): #set the threshold and the step 
                self.clust_num+=5
#                Lib_Cluster._build_clusters(self,self.clust_num,method)
#                print(self.clusters)
            if (abs(self.data.lb-self.lb[self.cut])>13072) and (self.clust_num>5) :
                if (self.lb[self.cut]>0):
                    self.clust_num+=-5
                else:
                    if self.clust_num<len(self.scenar)-5:
                        self.clust_num+=5
                    else:
                        self.clust_num+=0
            #return self.clust_num
#                Lib_Cluster._build_clusters(self,self.clust_num,method)
#                print(self.clusters)
##Modify the MIP gap
##            gp=0.00001
#            gp=(10e-6)+(abs(self.data.ub-self.data.lb)/abs(self.data.lb))/(len(self.data.cutlist)+10e-6)
##            print(gp)
#            self.model.setParam('MIPGap',gp)
#            self._do_benders_step(self.clust_num)
#        print('Objective Value: {0}'.format(self.model.Objval))
##        print('Power of the generators: {0}'.format(self.variables.Pgen))
##        print('Power of the winfarms: {0}'.format(self.variables.WindDA))
    

#INITIALIZE the data of the MP       
    def _init_benders_params(self,epsilon,delta,IterLimit,Exit,clust_num,scenar,proba,method):
        self.Resp_Fx={}      #initialisation of the storage elements for iterations
        self.Resn_Fx={}
        self.WindDA_Fx={}
        self.lineflowDA_Fx={}
        self.Objval={}
        self.sens_Resp={}  #initialisation of the storage elements for sensitivities
        self.sens_Resn={}
        self.sens_WindDA={}
        self.sens_lineflowDA={}
        self.Objval_agg=[] #initialisation of the aggregated storage object for objective value aggregation in single cut
        self.lb=[]
        self.ub=[]
        self.cuts=[2e-5]      #initialisation of cuts constraints for master problem
        self.data.cutlist=[]
        self.data.mipgap=[]
        self.data.IterLimit=IterLimit  #set a limit number of iteration
        self.data.delta=delta      #Absolute tolerance
        self.data.epsilon=epsilon   #relatave tolerance
        self.Exit=Exit
        self.data.ub=gb.GRB.INFINITY
        self.data.lb=-gb.GRB.INFINITY
        self.data.store_lshed={}
        self.data.store_wspill={}
        self.cut=0
        self.runtime_mp={}
        self.nzlist_mp={}
        self.cst_mp={}
        self.duals_cuts={}
        self.clust_num=clust_num
        self.remove_indic=0
        self.cut_use={} #indicator to count the succesive iteration where cuts are not used
        self.constraints_cuts_theta={}
        self.constraints_cuts_rhs={}
        self.store_binaryvar={}
        self.store_pgen={}
        self.store_resp={}
        self.store_resn={}
        self.runtime_sp=[]
        self.z_sub={}
        self.scenar=scenar
        self.proba=proba
        self.method=method
    
        
    def _load_data(self):
        Data_Load._load_network(self)     
        Data_Load._load_generator_data(self)
        Data_Load._load_wind_data(self)
        Data_Load._load_intial_data(self)
        
    def _build_model(self):
        self.model = gb.Model()        
        self.model.setParam('OutputFlag',0) #to have all the infos of optimization 
#        self.model.setParam('MIPGap',0.01)
        Lib_Variables_multiprocess_Parallel.build_variables_DA(self)
        Lib_Constraints_multiprocess_Parallel.build_masterpb_constr(self)
        #Build_constraints._build_constraints_DA(self)   
        Lib_ObjFnct_cluster_multiprocess_Parallel.build_objective_masterpb(self)             
        self.model.update()
        self.constraints.cuts={}
 
#function to create the clusters  
    def build_clusters(self):
#        print('build_clusters')
        clust_num=self.clust_num
##clustering using the objective value of sub-problem
#        #creation of the data set for clustering with all the obj val of the SPs
#        objval_list=[]
#        for s in range(1,NScen+1):
#            objval_list.append([self.submodels[s].model.Objval])
##        print(objval_list)
#        n_clusters=clust_num
#        kmeans = KMeans(n_clusters, random_state=0).fit(objval_list)
##        kmeans.labels_
#        self.clusters=[]
#        for n in range(n_clusters):
#            self.clusters.append([])
#        k=0
#        for i in range(NScen):
#            k+=1
#            self.clusters[kmeans.labels_[i]].append(k-1)
##clustering using the computed values (SP) wspill
#        #creation of the data
#        windfarms = self.data.windfarms 
#        self.wspill_list=[]
#        for s in range(1,NScen+1):
#            wspill_slist=[]
#            for j in windfarms:
#                for t in range(SchedulingHorizon):
#                    wspill_slist.append(m.submodels[s].variables.wspill[j][t][s].x)
#            self.wspill_list.append(wspill_slist)
#        n_clusters=clust_num
#        kmeans = KMeans(n_clusters, random_state=0).fit(self.wspill_list)
##        kmeans.labels_
#        self.clusters=[]
#        for n in range(n_clusters):
#            self.clusters.append([])
#        k=0
#        for i in range(NScen):
#            k+=1
#            self.clusters[kmeans.labels_[i]].append(k-1)
##clustering using the computed values (SP) powerDn/powerUp
#        #creation of the data
#        generators = self.data.generators
#        self.powerUp_list=[]
#        for s in range(1,NScen+1):
#            powerUp_slist=[]
#            for j in generators:
#                for t in range(SchedulingHorizon):
#                    powerUp_slist.append(m.submodels[s].variables.powerUp[j][t][s].x)
#            self.powerUp_list.append(powerUp_slist)
#        n_clusters=clust_num
#        kmeans = KMeans(n_clusters, random_state=0).fit(self.powerUp_list)
##        kmeans.labels_
#        self.clusters=[]
#        for n in range(n_clusters):
#            self.clusters.append([])
#        k=0
#        for i in range(NScen):
#            k+=1
#            self.clusters[kmeans.labels_[i]].append(k-1)            
##clustering using the computed values (SP) lshed
#        self.lshed_list=[]
#        for s in range(1,NScen+1):
#            lshed_slist=[]
#            for j in self.data.demand.index.tolist():
#                for t in range(SchedulingHorizon):
#                    lshed_slist.append(m.submodels[s].variables.lshed[j][t][s].x)
#            self.lshed_list.append(lshed_slist)
#        n_clusters=clust_num
#        kmeans = KMeans(n_clusters, random_state=0).fit(self.lshed_list)
##        kmeans.labels_
#        self.clusters=[]
#        for n in range(n_clusters):
#            self.clusters.append([])
#        k=0
#        for i in range(NScen):
#            k+=1
#            self.clusters[kmeans.labels_[i]].append(k-1)  
##clustering using the computed values (SP) duals
        generators=self.data.generators
        windfarms=self.data.windfarms
#        #get sensitivities from subproblem
        self.sens_Resp=sens_Resp
        self.sens_Resn=sens_Resn
        self.sens_WindDA=sens_WindDA
        self.sens_lineflowDA=sens_lineflowDA
        for s in self.scenar:
		#IF NOT PARALLELIZATION OF SP
##            print(self.submodels[s].subproblem_fixedp)
#            self.sens_Resp[s]={}
#            self.sens_Resn[s]={}
#            self.sens_WindDA[s]={}
#            self.sens_lineflowDA[s]={}
#            for g in generators:
#                self.sens_Resp[s][g]={
#                t: self.submodels[s].subproblem_fixedp[g][t].pi for t in range(SchedulingHorizon)}
#                self.sens_Resn[s][g]={
#                t: self.submodels[s].subproblem_fixedn[g][t].pi for t in range(SchedulingHorizon)}
#            for j in windfarms:
#                self.sens_WindDA[s][j]={
#                t: self.submodels[s].subproblem_fixedw[j][t].pi for t in range(SchedulingHorizon)}
#            for l in self.data.linesindex:
#                self.sens_lineflowDA[s][l]={
#                        t:self.submodels[s].subproblem_flow[l][t].pi for t in range(SchedulingHorizon)}
##Rescale these sensitivities to avoid numerical issues 
            for t in range(SchedulingHorizon):
                for i in self.data.generators:
                    if abs(self.sens_Resp[s][i][t])<=1e-8:
                        self.sens_Resp[s][i][t]=0
                    if abs(self.sens_Resn[s][i][t])<=1e-8:
                        self.sens_Resn[s][i][t]=0 
                for j in self.data.windfarms:
                    if abs(self.sens_WindDA[s][j][t])<=1e-8:
                        self.sens_WindDA[s][j][t]=0
                for l in self.data.linesindex:
                    if abs(self.sens_lineflowDA[s][l][t])<=1e-8:
                        self.sens_lineflowDA[s][l][t]=0
#normalisation and aggregation of these values ( to take the three duals as basis of the clustering)
        list_sens_Resp=[]
        list_sens_Resn=[]
        list_sens_WindDA=[]
        list_sens_lineflowDA=[]
        for s in self.scenar:
            for t in range(SchedulingHorizon):
                for i in self.data.generators:
                    list_sens_Resp.append(self.sens_Resp[s][i][t])
                    list_sens_Resn.append(self.sens_Resn[s][i][t])
                for j in self.data.windfarms:
                    list_sens_WindDA.append(self.sens_WindDA[s][j][t])
                for l in self.data.linesindex:
                   list_sens_lineflowDA.append(self.sens_lineflowDA[s][l][t])
        min_Resp=min(list_sens_Resp)
#        print(min_Resp)
        max_Resp=max(list_sens_Resp)
#        print(max_Resp)
        min_Resn=min(list_sens_Resn)
#        print(min_Resn)
        max_Resn=max(list_sens_Resn)
#        print(max_Resn)
        min_WindDA=min(list_sens_WindDA)
#        print(min_WindDA)
        max_WindDA=max(list_sens_WindDA)
#        print(max_WindDA)
        min_lineflowDA=min(list_sens_lineflowDA)
#        print(min_lineflowDA)
        max_lineflowDA=max(list_sens_lineflowDA)
#        print(max_lineflowDA)
        self.duals_list=[]
        for s in self.scenar:
            duals_slist=[]
            for i in generators:
                for t in range(SchedulingHorizon):
                    if (max_Resp-min_Resp) != 0:
                        duals_slist.append((self.sens_Resp[s][i][t]-min_Resp)/(max_Resp-min_Resp))
            for i in generators:    
                for t in range(SchedulingHorizon):
                    if (max_Resn-min_Resn) != 0:
                        duals_slist.append((self.sens_Resn[s][i][t]-min_Resn)/(max_Resn-min_Resn))
            for j in windfarms:
                for t in range(SchedulingHorizon):
                    if (max_WindDA-min_WindDA) != 0:
                        duals_slist.append((self.sens_WindDA[s][j][t]-min_WindDA)/(max_WindDA-min_WindDA))
            for l in self.data.linesindex:
                for t in range(SchedulingHorizon):
                    if (max_lineflowDA-min_lineflowDA) != 0:
                        duals_slist.append((self.sens_lineflowDA[s][l][t]-min_lineflowDA)/(max_lineflowDA-min_lineflowDA))
            self.duals_list.append(duals_slist)
#k-means clustering			
        if self.method=='k_means':
            n_clusters=self.clust_num
            kmeans = KMeans(n_clusters, random_state=0).fit(self.duals_list)
#            print(kmeans)
#            print(kmeans.labels_)
            self.clusters=[]
            for n in range(n_clusters):
                self.clusters.append([])
            k=0
            for i in range(NScen):
                k+=1
                self.clusters[kmeans.labels_[i]].append(k-1)
        if self.method=='k_medoids':
#k-medoids clustering
            n_clusters=self.clust_num
            D = pairwise_distances(np.array(self.duals_list), metric='euclidean')
            M, C = kmedoids.kMedoids(D, n_clusters) #M is a list of medoids and C a list of cluster (repartition of the scenarios in the clusters, only number (ID) of scenario not data)
#            print(M)
#            print(C)
            self.clusters=[]
            self.medoids=list(M)
            for i in range(n_clusters):
                self.clusters.append(list(C[i]))
        if self.method=='hierar':
#hierarchical clustering
            n_clusters=int(clust_num)
            cluster = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward').fit_predict(self.duals_list)  
            self.clusters=[]
            for n in range(n_clusters):
                self.clusters.append([])
            k=0
            for i in range(len(cluster)):
#                k+=1
#                self.clusters[cluster[i]].append(k-1) 
                self.clusters[cluster[i]].append(self.scenar[i])
        
        
    def _add_cut(self):
#        print('add cuts')
        generators=self.data.generators
#        generatorinfo=self.data.generatorinfo
        windfarms=self.data.windfarms
##former code        
#        self.cut=len(self.data.cutlist)
#        self.data.cutlist.append(self.cut)
#new code
        if len(self.data.cutlist)>=1:
            self.cut=self.data.cutlist[len(self.data.cutlist)-1]+1
        else:
            self.cut=len(self.data.cutlist)
        self.data.cutlist.append(self.cut)
        self.cut_use[self.cut]=0 #initialisation of the use marker
###############################################################################
#FORMER SOLUTIONS IF REMOVE
#remove the old cuts
# -2 not good solution        
#        if (self.cut-35)%8==0 and (self.cut-35)>=0:
#            print(self.cut)
#            for l in range(8):
#                for j in range(len(self.constraints.cuts[self.cut-35+l])):
#                    self.model.remove(self.model.getConstrByName('cut_{0}_scen_{1}'.format(self.cut-35+l,j)))
#
#new solution, more systematic
#        if self.cut>=1:
#            if self.runtime_mp[self.cut-1]>1.5:
#                if (self.cut-self.remove_indic)>5:
#                    for j in range(len(self.constraints.cuts[self.remove_indic])):
#                        self.model.remove(self.model.getConstrByName('cut_{0}_scen_{1}'.format(self.remove_indic,j)))
#                    self.data.cutlist.pop(0)
#                    self.remove_indic=self.remove_indic+1
####################################################################################
#UNCOMMENT FOR CUT CONSOLIDATION
##Cut consolidation
   #     if self.cut>1:
    #        #count the succesive iteration where cut is unuse
#    #        print(self.duals_cuts)
      #      for c in self.duals_cuts[len(self.duals_cuts)-1]:
       #         if all(abs(i)<10e-5 for i in self.duals_cuts[len(self.duals_cuts)-1][c]):
        #            self.cut_use[c]+=1
         #       else:
          #          if self.cut_use[c]<unused: #criterion should be defined
           #             self.cut_use[c]=0
            #        else:
             #           self.cut_use[c]+=1
              #  if self.cut_use[c]==unused:
               #     if (self.cut-c)>=unused:
                #        self.data.cutlist.remove(c)
                 #       for j in range(len(self.constraints.cuts[c])):
                  #          self.model.remove(self.model.getConstrByName('cut_{0}_scen_{1}'.format(c,j))) #remove extensive form of cuts
#                  #          print(c,j)
                    #    self.model.addConstr(gb.quicksum(self.constraints_cuts_theta[c][j] for j in range(len(self.constraints_cuts_theta[c]))),gb.GRB.GREATER_EQUAL,
                     #                        gb.quicksum(self.constraints_cuts_rhs[c][j] for j in range(len(self.constraints_cuts_rhs[c]))),name='cut_consolidation{0}'.format(c))
                    
                    
                    
                
###############################################################################            
#CREATE THE CUTS       
        self.constraints.cuts[self.cut]={}
        self.constraints_cuts_theta[self.cut]={}
        self.constraints_cuts_rhs[self.cut]={}
#        print('clusters')
#        print(self.clusters)
#        print(self.data.scenprob)
#        print(self.sens_Resn)
        for j in range(len(self.clusters)):
            self.constraints.cuts[self.cut][j]=self.model.addConstr(gb.quicksum(self.proba[s+1]*self.variables.theta[s] for s in self.clusters[j]),gb.GRB.GREATER_EQUAL,
                                 gb.quicksum((self.proba[s+1]*(self.z_sub[s]))for s in self.clusters[j])+gb.quicksum(self.proba[s+1]*self.sens_Resp[s][i][t]*(self.variables.Resp[i][t]-self.Resp_Fx[i][t])for i in generators for t in range(SchedulingHorizon) for s in self.clusters[j])
                                 +(gb.quicksum(self.proba[s+1]*self.sens_Resn[s][i][t]*(self.variables.Resn[i][t]-self.Resn_Fx[i][t])for i in generators for t in range(SchedulingHorizon) for s in self.clusters[j]))
                                 +(gb.quicksum(self.proba[s+1]*self.sens_WindDA[s][i][t]*(self.variables.WindDA[i][t]-self.WindDA_Fx[i][t])for i in windfarms for t in range(SchedulingHorizon) for s in self.clusters[j]))
                                 +(gb.quicksum(self.proba[s+1]*self.sens_lineflowDA[s][l][t]*(self.variables.lineflowDA[l][t]-self.lineflowDA_Fx[l][t]) for l in self.data.linesindex for t in range(SchedulingHorizon) for s in self.clusters[j])),name='cut_{0}_scen_{1}'.format(self.cut,j))
#store data for cut consolidation (to be able to aggregate later the cuts)
            self.constraints_cuts_theta[self.cut][j]=gb.quicksum(self.proba[s+1]*self.variables.theta[s] for s in self.clusters[j])
            self.constraints_cuts_rhs[self.cut][j]= (gb.quicksum((self.proba[s+1]*(self.z_sub[s]))for s in self.clusters[j])+gb.quicksum(self.proba[s+1]*self.sens_Resp[s][i][t]*(self.variables.Resp[i][t]-self.Resp_Fx[i][t])for i in generators for t in range(SchedulingHorizon) for s in self.clusters[j])
                                 +(gb.quicksum(self.proba[s+1]*self.sens_Resn[s][i][t]*(self.variables.Resn[i][t]-self.Resn_Fx[i][t])for i in generators for t in range(SchedulingHorizon) for s in self.clusters[j]))
                                 +(gb.quicksum(self.proba[s+1]*self.sens_WindDA[s][i][t]*(self.variables.WindDA[i][t]-self.WindDA_Fx[i][t])for i in windfarms for t in range(SchedulingHorizon) for s in self.clusters[j]))
                                 +(gb.quicksum(self.proba[s+1]*self.sens_lineflowDA[s][l][t]*(self.variables.lineflowDA[l][t]-self.lineflowDA_Fx[l][t]) for l in self.data.linesindex for t in range(SchedulingHorizon) for s in self.clusters[j])))


        
#UPDATE LOWER AND UPPER BOUND
    def _update_bounds(self,Sub_results):
#        print('update')
        self.sens_Resp=Sub_results[0]
        self.sens_Resn=Sub_results[1]
        self.sens_WindDA=Sub_results[2]
        self.sens_lineflowDA=Sub_results[3]
        self.z_sub=Sub_results[4]
#        print(self.sens_Resp,self.sens_Resn)
#        print(z_master)
        self.z_master=self.model.Objval
#        print(self.z_master)
#        z_sub_sum=sum(self.submodels[s].model.Objval*self.data.scenprob[s] for s in range(1,NScen+1))
#        print(self.z_sub)
        z_sub_sum=sum(self.z_sub[s]*self.proba[s+1] for s in self.scenar)
#        print(z_sub_sum)
#        print(z_sub_sum)
#        for s in range(NScen):
#            print(self.variables.theta[s].x)
#        if multicut==1:
##            print(z_master)
##            print(z_sub)
##            print(sum(self.data.scenprob[s]*self.variables.theta[s].x for s in range(NScen)))
#            self.data.ub=z_master+z_sub_sum-sum(self.data.scenprob[s]*self.variables.theta[s].x for s in range(1,NScen+1))
#        else:
        self.data.ub=self.z_master+z_sub_sum-sum(self.proba[s+1]*self.variables.theta[s].x for s in self.scenar)  #self.data.scenprob[s]*
        self.data.lb=self.model.ObjVal
        self.lb.append(self.data.lb)
        self.ub.append(self.data.ub)
        self.data.mipgap.append(self.model.params.IntFeasTol)
#check termination criteria
#        print(self.data.ub)
#        print(self.data.lb)
        if (abs(self.data.ub-self.data.lb) < self.data.delta):
#            print("Benders step : {0}*************************************************************".format(len(self.data.cutlist)))
#            print("diff*******************************{0}".format(self.data.ub-self.data.lb))
#            print("Upper bound:{0}".format(self.data.ub))
#            print("Lower bound :{}".format(self.data.lb))
            self.Exit=1
            print('Benders termination:Absolute tolerance reached')
        elif (abs(self.data.ub-self.data.lb) < (self.data.epsilon*self.data.lb)):
            self.Exit=0
#            print('Benders termination: Relative tolerance reached')
        elif (len(self.data.cutlist) > self.data.IterLimit):
            self.Exit=1
            print('Benders termination: Iteration Limit reached')
        else:
#            print("Benders step : {0}*************************************************************".format(len(self.data.cutlist)))
#            print("diff*******************************{0}".format(self.data.ub-self.data.lb))
#            print("Upper bound:{0}".format(self.data.ub))
#            print("Lower bound :{}".format(self.data.lb))
            self.Exit=0
        return(self.Exit)
    
    def _save_vars(self):
#        if multicut==1:
#            c=[]
#            for s in range(1,NScen+1):
#                c.append(self.variables.theta[s].x)
#            self.Objval_agg.append(c)
#        else:
        self.Objval_agg.append(gb.quicksum(self.variables.WindDA[j][0].x for j in self.data.windfarms))
#        print(clust_num)
        fixed=self.model.fixed()
#        fixed.Params.LogToConsole=0
        fixed.optimize()   
############## if all cuts  ########################## 
#        self.duals_cuts[len(self.data.cutlist)]={}
#        for c in range(len(self.data.cutlist)):
#            self.duals_cuts[len(self.data.cutlist)][c]=[]
#            for s in range(len(self.constraints.cuts[c])):
#                self.duals_cuts[len(self.data.cutlist)][c].append(fixed.getConstrByName('cut_{0}_scen_{1}'.format(c,s)).pi)
############# if remove some cuts ##############################
#UNCOMMENT FOR CUT CONSOLIDATION
#        print(self.data.cutlist)
#        print(self.duals_cuts)
    #    self.duals_cuts[self.data.cutlist[len(self.data.cutlist)-1]]={}
     #   for c in self.data.cutlist:
      #      self.duals_cuts[self.data.cutlist[len(self.data.cutlist)-1]][c]=[]
       #     for s in range(len(self.constraints.cuts[c])):
        #        self.duals_cuts[self.data.cutlist[len(self.data.cutlist)-1]][c].append(fixed.getConstrByName('cut_{0}_scen_{1}'.format(c,s)).pi)
        

###################################################################################################
###########################################RUN THE BENDERS PROCESS############################### 
sens_Resp={}
sens_Resn={}
sens_WindDA={}
sens_lineflowDA={}
z_sub={}
Resp={}
Resn={}
WindDA={}
lineflowDA={}
unused=5 #Define the threshold for the CUT CONSOLIDATION
###################################################################################################
# FIX THE VALUES FOR THE SECOND STEP
#FIX={1: {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 2: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 3: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 4: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 5: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 6: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 7: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 8: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 9: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 10: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 11: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0}, 12: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 13: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 14: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 15: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 16: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 17: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 18: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 19: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 20: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 21: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 22: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 23: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 24: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 25: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 26: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 27: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 28: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 29: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 30: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 31: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 32: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 33: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 34: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 35: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 36: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 37: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 38: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 39: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 40: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 41: {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0}, 42: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 43: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 44: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 45: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 46: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 47: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 48: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 49: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 50: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 51: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 52: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 53: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 54: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 55: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 56: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 57: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 58: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 59: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 60: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 61: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 62: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 11: 0.0}, 63: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 11: 0.0}, 64: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 65: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 66: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 67: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 68: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 69: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 70: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 71: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 72: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 73: {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 74: {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 75: {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 76: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 77: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 78: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 79: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 80: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 81: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 82: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 83: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 84: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0}, 85: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0}, 86: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 87: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 88: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 89: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 90: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 91: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 92: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 9: 1.0, 10: 1.0, 11: 0.0}, 93: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 94: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 95: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 96: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}}
#FIX={1: {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 2: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 3: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 4: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 5: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 6: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 7: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 8: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 9: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 10: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 11: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0}, 12: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 13: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 14: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 15: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 16: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 17: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 18: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 19: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 20: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 21: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 22: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 23: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 24: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 25: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 26: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 27: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 28: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 29: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 30: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 31: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 32: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 33: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 34: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 35: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 36: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 37: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 38: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 39: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 40: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 41: {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0}, 42: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 43: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 44: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 45: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 46: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 47: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 48: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 49: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 50: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 51: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 52: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 53: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 54: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 55: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 56: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 57: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 58: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 59: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 60: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 61: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 62: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 11: 0.0}, 63: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 11: 0.0}, 64: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 65: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 66: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 67: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 68: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 69: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 70: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 71: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 72: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 73: {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 74: {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 75: {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 76: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 77: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 78: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 79: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 80: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 81: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 82: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 83: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 84: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 85: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 86: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 87: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 88: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 89: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 90: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 91: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 92: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 9: 1.0, 10: 1.0, 11: 0.0}, 93: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 94: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 95: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 96: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}}
#fix for new gen and adm new
#FIX={1: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 2: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 3: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 4: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 5: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 6: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 7: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 8: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 9: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 10: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 11: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 12: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 13: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 14: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 15: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 16: {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 17: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 18: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 19: {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 8: 1.0, 9: 1.0, 10: 1.0}, 20: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 21: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 22: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 23: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 24: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 25: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 26: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 27: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 28: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 29: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 30: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 31: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 32: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 33: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 34: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 35: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 36: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 37: {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 38: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 39: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 40: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 41: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 42: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 43: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 44: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 11: 0.0}, 45: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 46: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}}
#fix for Benders
#FIX={1: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 2: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 3: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 4: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 5: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 6: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 7: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 8: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 9: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 10: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 11: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 12: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 13: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 14: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 15: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 16: {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 17: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 18: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 19: {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 20: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 21: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 22: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 23: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 24: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 25: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 26: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 27: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 28: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 29: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 30: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 31: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 32: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 33: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 34: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 35: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 36: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 37: {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 38: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 39: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 40: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 41: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 42: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 43: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 1.0, 9: 1.0, 10: 0.0, 11: 0.0}, 44: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 1.0, 9: 1.0, 10: 0.0, 11: 0.0}, 45: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 46: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}}
#fix for no benders
#FIX={1: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: -0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 2: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: -0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 3: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 4: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: -0.0, 6: -0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 5: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 6: {0: 0.0, 1: -0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 7: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 8: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 9: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 10: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 11: {0: 0.0, 1: 0.0, 2: 0.0, 3: -0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: -0.0, 10: 0.0, 11: 0.0}, 12: {0: 0.0, 1: -0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: -0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 13: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 14: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: -0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 15: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: -0.0, 10: 0.0, 11: 0.0}, 16: {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: -0.0}, 17: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: -0.0, 9: 0.0, 10: 0.0, 11: -0.0}, 18: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 19: {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 20: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: -0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 21: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 22: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: -0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 23: {0: 0.0, 1: 0.0, 2: 0.0, 3: -0.0, 4: 0.0, 5: 0.0, 6: -0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 24: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 25: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 26: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 27: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: -0.0, 7: 0.0, 8: 0.0, 9: -0.0, 10: 0.0, 11: 0.0}, 28: {0: 0.0, 1: 0.0, 2: -0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: -0.0, 11: 0.0}, 29: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 30: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: -0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 31: {0: -0.0, 1: 0.0, 2: -0.0, 3: -0.0, 4: 0.0, 5: -0.0, 6: -0.0, 7: 0.0, 8: -0.0, 9: 0.0, 10: -0.0, 11: 0.0}, 32: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 33: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: -0.0, 7: 0.0, 8: -0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 34: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 35: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: -0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 36: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: -0.0, 5: -0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 37: {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: -0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 38: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: -0.0, 7: -0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: -0.0}, 39: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: -0.0, 6: 0.0, 7: 0.0, 8: -0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 40: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: -0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 41: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}, 42: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 43: {0: -0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: -0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 44: {0: 0.0, 1: 0.0, 2: 0.0, 3: -0.0, 4: 0.0, 5: -0.0, 6: 0.0, 7: 0.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 0.0}, 45: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0}, 46: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: -0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}}
#fix for benders 10 clusters
#FIX={1: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 2: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 22: 0.0, 23: 0.0}, 3: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 0.0, 23: 0.0}, 4: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 5: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 23: 0.0}, 6: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 7: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 8: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 9: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 10: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 11: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 12: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 13: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 14: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 15: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 16: {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 17: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 18: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 19: {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 20: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 21: 0.0, 22: 0.0, 23: 0.0}, 21: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 0.0, 22: 0.0, 23: 0.0}, 22: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 18: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 23: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 24: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 25: {4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 26: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 27: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 28: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 29: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 30: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 31: {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 32: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 33: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 34: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 35: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 0.0, 23: 0.0}, 36: {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0}, 37: {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 38: {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 10: 0.0, 14: 0.0, 15: 0.0, 16: 1.0, 17: 1.0, 18: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 39: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 40: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 41: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 42: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 43: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 44: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 45: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 22: 0.0, 23: 0.0}, 46: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13:0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}}
#fix for 30 clusters
#FIX={1: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 2: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 3: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 0.0, 23: 0.0}, 4: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 5: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 0.0}, 6: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 7: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 8: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 9: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 10: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 11: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 12: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 13: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 14: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 15: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 16: {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 17: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 18: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 19: {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 20: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 0.0, 23: 0.0}, 21: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 0.0, 22: 0.0, 23: 0.0}, 22: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 10: 0.0, 11: 0.0, 14: 0.0, 15: 0.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 23: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 24: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 25: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 26: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 27: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 28: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 29: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 30: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 31: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 32: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 33: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 34: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 35: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 0.0, 23: 0.0}, 36: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 37: {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 38: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 10: 0.0, 12: 1.0, 13: 1.0, 14: 0.0, 15: 0.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 39: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 40: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 41: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 42: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 43: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 18: 1.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 44: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 45: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 0.0, 23: 0.0}, 46: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}}
#Fix 30 clusters 85%
#FIX={1: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 2: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 3: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 0.0, 23: 0.0}, 4: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 5: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 0.0}, 6: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 7: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 8: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 9: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 10: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 11: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 12: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 13: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 14: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 15: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 16: {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 17: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 18: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 19: {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 20: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 0.0, 23: 0.0}, 21: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 0.0, 22: 0.0, 23: 0.0}, 22: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 10: 0.0, 11: 0.0, 14: 0.0, 15: 0.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 23: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 24: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 25: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 26: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 27: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 28: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 29: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 30: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 31: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 32: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 33: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 34: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 35: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 0.0, 23: 0.0}, 36: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 37: {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 38: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 10: 0.0, 12: 1.0, 13: 1.0, 14: 0.0, 15: 0.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 39: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 40: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 41: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 42: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 43: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 18: 1.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 44: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 45: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 0.0, 23: 0.0}, 46: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}}
#Fix 30 clusters 75%
#FIX={1: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 2: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 3: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 0.0, 23: 0.0}, 4: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 5: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 0.0}, 6: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 7: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 8: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 9: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 10: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 11: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 12: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 13: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 14: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 15: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 16: {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 17: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 18: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 19: {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 20: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 0.0, 23: 0.0}, 21: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 0.0, 22: 0.0, 23: 0.0}, 22: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 10: 0.0, 11: 0.0, 14: 0.0, 15: 0.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 23: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 24: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 25: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 26: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 27: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 28: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 29: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 30: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 31: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 32: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 33: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 34: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 35: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 0.0, 23: 0.0}, 36: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 37: {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 38: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 10: 0.0, 12: 1.0, 13: 1.0, 14: 0.0, 15: 0.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 39: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 40: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 41: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 42: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0}, 43: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 44: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 1.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}, 45: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 0.0, 23: 0.0}, 46: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0}}
######################################################
#Process for Benders + improvements
def SUCperCluster(L,i):
    scenar=L[0][i]
    proba=L[1][i]
    print(scenar)
    print(proba)
    store_time_MP=[]
    store_time_SP=[]
    Counter=0
    t0=time()
#FIX INITIAL NUMBER OF CLUSTER 
    #clust_num=0.6666666*len(scenar)
    #clust_num=len(scenar)
    clust_num=1
    method='hierar' #fix the clustering method
    m=MasterProblem(clust_num,method,scenar,proba) #create MP
	#######FIX variables values after first round of SUC (UNCOMMENT IN SECOND STEP)###########################
 #   for i in range(1,len(FIX)+1):
  #      for j in FIX[i]:
   #         m.variables.Stat[i][j].lb=FIX[i][j]
    #        m.variables.Stat[i][j].ub=FIX[i][j]
#    ##############################WARM START################################################
    #for i in range(1,len(FIX)+1):
     #   for j in FIX[i]:
      #      m.variables.Stat[i][j].start=FIX[i][j]
##################################################################################
    m.model.Params.MIPGap=1e-6 #define MIP Gap
    v=m.optimize(clust_num)
    t2=time()
    store_time_MP.append(t2-t0)
    SP_results=run_sp_parallel(scenar,v[1],v[0],v[2],v[3],-1)#-1 is the number of jobs (all if -1)
    t3=time()
    store_time_SP.append(t3-t2)
    Exit=m._update_bounds(SP_results)
#    m._save_vars()
#    print(m.data.lb,m.data.ub)
    while Exit!=1:
        t4=time()
        Counter+=1
        #print(Counter)
        m._change_clust_num()
        #print(m.data.lb)
        #print(m.lb[m.cut])
        #print(abs(m.data.lb-m.lb[m.cut]))
        print(m.clust_num)
        m.build_clusters()
        m._add_cut()
        m.model.update()
        #    m=MasterProblem(clust_num,method)
        v=m.optimize(m.clust_num)
        #print(m.model.Status)
        t5=time()
        store_time_MP.append(t5-t4)
        SP_results=run_sp_parallel(scenar,v[1],v[0],v[2],v[3],-1)#-1 is the nb of jobs
        t6=time()
        store_time_SP.append(t6-t5)
        Exit=m._update_bounds(SP_results)
        m._save_vars()
#        print(m.data.lb,m.data.ub)
#        print(Exit)
#    print('Objective Value: {0}'.format(m.model.Objval))   
    UC={}
    StrtUP={}
    StrtDN={}
    PW={}
    RESP={}
    RESN={}
    WND={}
    FLW={}
    DLT={}
    for g in m.data.generators:
        UC[g]={t:m.variables.Stat[g][t].x for t in range(SchedulingHorizon)}
        StrtUP[g]={t:m.variables.Stup[g][t].x for t in range(SchedulingHorizon)}
        StrtDN[g]={t:m.variables.Shdwn[g][t].x for t in range(SchedulingHorizon)}
        PW[g]={t:m.variables.Pgen[g][t].x for t in range(SchedulingHorizon)}
        RESP[g]={t:m.variables.Resp[g][t].x for t in range(SchedulingHorizon)}
        RESN[g]={t:m.variables.Resn[g][t].x for t in range(SchedulingHorizon)}
    for w in m.data.windfarms:
        WND[w]={t:m.variables.WindDA[w][t].x for t in range(SchedulingHorizon)}
    for l in m.data.linesindex:
        FLW[l]={t:m.variables.lineflowDA[l][t].x for t in range(SchedulingHorizon)}
    t1=time()
    C=m.model.Status
#    T_kmean=t1-t0
#    print(T_kmean)
#    return(m.model.Objval,m.variables.Pgen,m.variables.Stat,m.variables.Stup,m.variables.Shdwn,t1-t0) 
    return(m.model.Objval,t1-t0,m.model.NumConstrs,m.model.NumVars,C,Counter,UC,StrtUP,StrtDN,PW,RESP,RESN,WND,FLW)

   
#######################################################################################
#CODE FOR SERVER + CLUSTERS USED IN THE SERVERS (in the server only index given but data base of clusters is given here )
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
	#Clusters for different parameters
    #with 50 clusters
    #L=[[[2, 48, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [23, 43, 60, 2, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [7, 32, 87, 93, 2, 23, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [9, 2, 23, 7, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [10, 66, 2, 23, 7, 9, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [0, 11, 2, 23, 7, 9, 10, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [14, 2, 23, 7, 9, 10, 0, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [15, 2, 23, 7, 9, 10, 0, 14, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [19, 28, 2, 23, 7, 9, 10, 0, 14, 15, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [20, 2, 23, 7, 9, 10, 0, 14, 15, 19, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [21, 59, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [24, 95, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [64, 71, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [13, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [27, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [29, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [31, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [34, 41, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [36, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [37, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [40, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [17, 38, 44, 50, 56, 82, 84, 89, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [1, 65, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [46, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [47, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [45, 49, 97, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [51, 78, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [52, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [3, 4, 5, 26, 33, 54, 68, 86, 91, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [55, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [18, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [57, 72, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [61, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [8, 22, 25, 53, 62, 80, 90, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [39, 63, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [12, 69, 92, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [30, 70, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [73, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [75, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [77, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 76, 79, 81, 85, 35, 16, 94, 42, 98, 58], [76, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 79, 81, 85, 35, 16, 94, 42, 98, 58], [79, 83, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 81, 85, 35, 16, 94, 42, 98, 58], [81, 88, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 85, 35, 16, 94, 42, 98, 58], [85, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 35, 16, 94, 42, 98, 58], [35, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 16, 94, 42, 98, 58], [16, 74, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 94, 42, 98, 58], [94, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 42, 98, 58], [6, 42, 96, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 98, 58], [98, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 58], [58, 67, 99, 2, 23, 7, 9, 10, 0, 14, 15, 19, 20, 21, 24, 64, 13, 27, 29, 31, 34, 36, 37, 40, 44, 1, 46, 47, 49, 51, 52, 68, 55, 18, 57, 61, 62, 39, 69, 30, 73, 75, 77, 76, 79, 81, 85, 35, 16, 94, 42, 98]], {0: {3: 0.01, 49: 0.01, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 1: {24: 0.01, 44: 0.01, 61: 0.01, 3: 0.02, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 2: {8: 0.01, 33: 0.01, 88: 0.01, 94: 0.01, 3: 0.02, 24: 0.03, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 3: {10: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 4: {11: 0.01, 67: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 5: {1: 0.01, 12: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 6: {15: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 7: {16: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 8: {20: 0.01, 29: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 9: {21: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 10: {22: 0.01, 60: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 11: {25: 0.01, 96: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 12: {65: 0.01, 72: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 13: {14: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 14: {28: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 15: {30: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 16: {32: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 17: {35: 0.01, 42: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 18: {37: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 19: {38: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 20: {41: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 21: {18: 0.01, 39: 0.01, 45: 0.01, 51: 0.01, 57: 0.01, 83: 0.01, 85: 0.01, 90: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 22: {2: 0.01, 66: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 23: {47: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 24: {48: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 25: {46: 0.01, 50: 0.01, 98: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 26: {52: 0.01, 79: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 27: {53: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 28: {4: 0.01, 5: 0.01, 6: 0.01, 27: 0.01, 34: 0.01, 55: 0.01, 69: 0.01, 87: 0.01, 92: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 29: {56: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 30: {19: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 31: {58: 0.01, 73: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 32: {62: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 33: {9: 0.01, 23: 0.01, 26: 0.01, 54: 0.01, 63: 0.01, 81: 0.01, 91: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 34: {40: 0.01, 64: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 35: {13: 0.01, 70: 0.01, 93: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 36: {31: 0.01, 71: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 37: {74: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 38: {76: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 39: {78: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 40: {77: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 41: {80: 0.01, 84: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 42: {82: 0.01, 89: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 43: {86: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 44: {36: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 45: {17: 0.01, 75: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 95: 0.01, 43: 0.03, 99: 0.01, 59: 0.03}, 46: {95: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 43: 0.03, 99: 0.01, 59: 0.03}, 47: {7: 0.01, 43: 0.01, 97: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 99: 0.01, 59: 0.03}, 48: {99: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 59: 0.03}, 49: {59: 0.01, 68: 0.01, 100: 0.01, 3: 0.02, 24: 0.03, 8: 0.04, 10: 0.01, 11: 0.02, 1: 0.02, 15: 0.01, 16: 0.01, 20: 0.02, 21: 0.01, 22: 0.02, 25: 0.02, 65: 0.02, 14: 0.01, 28: 0.01, 30: 0.01, 32: 0.01, 35: 0.02, 37: 0.01, 38: 0.01, 41: 0.01, 45: 0.08, 2: 0.02, 47: 0.01, 48: 0.01, 50: 0.03, 52: 0.02, 53: 0.01, 69: 0.09, 56: 0.01, 19: 0.01, 58: 0.02, 62: 0.01, 63: 0.07, 40: 0.02, 70: 0.03, 31: 0.02, 74: 0.01, 76: 0.01, 78: 0.01, 77: 0.01, 80: 0.02, 82: 0.02, 86: 0.01, 36: 0.01, 17: 0.02, 95: 0.01, 43: 0.03, 99: 0.01}}]
	#with 30 clusters
    #L=[[[0, 8, 16, 29, 45, 46, 49, 50, 53, 60, 65, 72, 78, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [7, 18, 85, 97, 8, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [21, 8, 7, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [27, 83, 8, 7, 21, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [19, 28, 8, 7, 21, 27, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [32, 8, 7, 21, 27, 19, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [33, 54, 8, 7, 21, 27, 19, 32, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [35, 8, 7, 21, 27, 19, 32, 33, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [11, 8, 7, 21, 27, 19, 32, 33, 35, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [24, 38, 56, 8, 7, 21, 27, 19, 32, 33, 35, 11, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [43, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [47, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [6, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [41, 52, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [20, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [30, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [59, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [1, 3, 13, 55, 61, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [25, 36, 62, 94, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [9, 73, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 22, 74, 57, 77, 76, 80, 86, 87, 96, 99], [22, 40, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 74, 57, 77, 76, 80, 86, 87, 96, 99], [10, 14, 34, 64, 71, 74, 81, 90, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 57, 77, 76, 80, 86, 87, 96, 99], [57, 98, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 77, 76, 80, 86, 87, 96, 99], [77, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 76, 80, 86, 87, 96, 99], [76, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 80, 86, 87, 96, 99], [80, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 86, 87, 96, 99], [2, 4, 5, 12, 15, 17, 23, 31, 37, 39, 44, 48, 51, 58, 63, 66, 67, 69, 75, 79, 82, 86, 88, 89, 91, 92, 95, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 87, 96, 99], [26, 68, 84, 87, 93, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 96, 99], [42, 70, 96, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 99], [99, 8, 7, 21, 27, 19, 32, 33, 35, 11, 38, 43, 47, 6, 41, 20, 30, 59, 61, 25, 9, 22, 74, 57, 77, 76, 80, 86, 87, 96]], {0: {1: 0.01, 9: 0.01, 17: 0.01, 30: 0.01, 46: 0.01, 47: 0.01, 50: 0.01, 51: 0.01, 54: 0.01, 61: 0.01, 66: 0.01, 73: 0.01, 79: 0.01, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 1: {8: 0.01, 19: 0.01, 86: 0.01, 98: 0.01, 9: 0.12999999999999998, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 2: {22: 0.01, 9: 0.12999999999999998, 8: 0.04, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 3: {28: 0.01, 84: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 4: {20: 0.01, 29: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 5: {33: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 6: {34: 0.01, 55: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 7: {36: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 8: {12: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 9: {25: 0.01, 39: 0.01, 57: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 10: {44: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 11: {48: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 12: {7: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 13: {42: 0.01, 53: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 14: {21: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 15: {31: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 16: {60: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 17: {2: 0.01, 4: 0.01, 14: 0.01, 56: 0.01, 62: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 18: {26: 0.01, 37: 0.01, 63: 0.01, 95: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 19: {10: 0.01, 74: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 20: {23: 0.01, 41: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 21: {11: 0.01, 15: 0.01, 35: 0.01, 65: 0.01, 72: 0.01, 75: 0.01, 82: 0.01, 91: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 22: {58: 0.01, 99: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 23: {78: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 24: {77: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 25: {81: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03, 100: 0.01}, 26: {3: 0.01, 5: 0.01, 6: 0.01, 13: 0.01, 16: 0.01, 18: 0.01, 24: 0.01, 32: 0.01, 38: 0.01, 40: 0.01, 45: 0.01, 49: 0.01, 52: 0.01, 59: 0.01, 64: 0.01, 67: 0.01, 68: 0.01, 70: 0.01, 76: 0.01, 80: 0.01, 83: 0.01, 87: 0.01, 89: 0.01, 90: 0.01, 92: 0.01, 93: 0.01, 96: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 88: 0.05, 97: 0.03, 100: 0.01}, 27: {27: 0.01, 69: 0.01, 85: 0.01, 88: 0.01, 94: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 97: 0.03, 100: 0.01}, 28: {43: 0.01, 71: 0.01, 97: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 100: 0.01}, 29: {100: 0.01, 9: 0.12999999999999998, 8: 0.04, 22: 0.01, 28: 0.02, 20: 0.02, 33: 0.01, 34: 0.02, 36: 0.01, 12: 0.01, 39: 0.03, 44: 0.01, 48: 0.01, 7: 0.01, 42: 0.02, 21: 0.01, 31: 0.01, 60: 0.01, 62: 0.05, 26: 0.04, 10: 0.02, 23: 0.02, 75: 0.08, 58: 0.02, 78: 0.01, 77: 0.01, 81: 0.01, 87: 0.2700000000000001, 88: 0.05, 97: 0.03}}]
	#with 10 clusters
    #L=[[[3, 4, 44, 50, 56, 68, 91, 92, 99, 64, 66, 24, 1, 7, 27, 89, 94, 42], [64, 71, 4, 66, 24, 1, 7, 27, 89, 94, 42], [2, 5, 8, 10, 11, 12, 13, 14, 16, 17, 20, 22, 26, 30, 35, 36, 38, 40, 41, 43, 45, 48, 49, 51, 52, 53, 55, 58, 59, 61, 62, 63, 66, 67, 69, 70, 73, 74, 75, 77, 80, 81, 82, 86, 88, 4, 64, 24, 1, 7, 27, 89, 94, 42], [24, 90, 95, 4, 64, 66, 1, 7, 27, 89, 94, 42], [1, 65, 4, 64, 66, 24, 7, 27, 89, 94, 42], [6, 7, 18, 32, 33, 37, 39, 47, 79, 84, 85, 87, 93, 97, 4, 64, 66, 24, 1, 27, 89, 94, 42], [27, 83, 4, 64, 66, 24, 1, 7, 89, 94, 42], [0, 15, 19, 21, 23, 28, 29, 54, 60, 76, 78, 89, 4, 64, 66, 24, 1, 7, 27, 94, 42], [9, 25, 94, 4, 64, 66, 24, 1, 7, 27, 89, 42], [31, 34, 42, 46, 57, 72, 96, 98, 4, 64, 66, 24, 1, 7, 27, 89, 94]], {0: {4: 0.01, 5: 0.01, 45: 0.01, 51: 0.01, 57: 0.01, 69: 0.01, 92: 0.01, 93: 0.01, 100: 0.01, 65: 0.02, 67: 0.45000000000000023, 25: 0.03, 2: 0.02, 8: 0.13999999999999999, 28: 0.02, 90: 0.11999999999999998, 95: 0.03, 43: 0.08}, 1: {65: 0.01, 72: 0.01, 5: 0.09, 67: 0.45000000000000023, 25: 0.03, 2: 0.02, 8: 0.13999999999999999, 28: 0.02, 90: 0.11999999999999998, 95: 0.03, 43: 0.08}, 2: {3: 0.01, 6: 0.01, 9: 0.01, 11: 0.01, 12: 0.01, 13: 0.01, 14: 0.01, 15: 0.01, 17: 0.01, 18: 0.01, 21: 0.01, 23: 0.01, 27: 0.01, 31: 0.01, 36: 0.01, 37: 0.01, 39: 0.01, 41: 0.01, 42: 0.01, 44: 0.01, 46: 0.01, 49: 0.01, 50: 0.01, 52: 0.01, 53: 0.01, 54: 0.01, 56: 0.01, 59: 0.01, 60: 0.01, 62: 0.01, 63: 0.01, 64: 0.01, 67: 0.01, 68: 0.01, 70: 0.01, 71: 0.01, 74: 0.01, 75: 0.01, 76: 0.01, 78: 0.01, 81: 0.01, 82: 0.01, 83: 0.01, 87: 0.01, 89: 0.01, 5: 0.09, 65: 0.02, 25: 0.03, 2: 0.02, 8: 0.13999999999999999, 28: 0.02, 90: 0.11999999999999998, 95: 0.03, 43: 0.08}, 3: {25: 0.01, 91: 0.01, 96: 0.01, 5: 0.09, 65: 0.02, 67: 0.45000000000000023, 2: 0.02, 8: 0.13999999999999999, 28: 0.02, 90: 0.11999999999999998, 95: 0.03, 43: 0.08}, 4: {2: 0.01, 66: 0.01, 5: 0.09, 65: 0.02, 67: 0.45000000000000023, 25: 0.03, 8: 0.13999999999999999, 28: 0.02, 90: 0.11999999999999998, 95: 0.03, 43: 0.08}, 5: {7: 0.01, 8: 0.01, 19: 0.01, 33: 0.01, 34: 0.01, 38: 0.01, 40: 0.01, 48: 0.01, 80: 0.01, 85: 0.01, 86: 0.01, 88: 0.01, 94: 0.01, 98: 0.01, 5: 0.09, 65: 0.02, 67: 0.45000000000000023, 25: 0.03, 2: 0.02, 28: 0.02, 90: 0.11999999999999998, 95: 0.03, 43: 0.08}, 6: {28: 0.01, 84: 0.01, 5: 0.09, 65: 0.02, 67: 0.45000000000000023, 25: 0.03, 2: 0.02, 8: 0.13999999999999999, 90: 0.11999999999999998, 95: 0.03, 43: 0.08}, 7: {1: 0.01, 16: 0.01, 20: 0.01, 22: 0.01, 24: 0.01, 29: 0.01, 30: 0.01, 55: 0.01, 61: 0.01, 77: 0.01, 79: 0.01, 90: 0.01, 5: 0.09, 65: 0.02, 67: 0.45000000000000023, 25: 0.03, 2: 0.02, 8: 0.13999999999999999, 28: 0.02, 95: 0.03, 43: 0.08}, 8: {10: 0.01, 26: 0.01, 95: 0.01, 5: 0.09, 65: 0.02, 67: 0.45000000000000023, 25: 0.03, 2: 0.02, 8: 0.13999999999999999, 28: 0.02, 90: 0.11999999999999998, 43: 0.08}, 9: {32: 0.01, 35: 0.01, 43: 0.01, 47: 0.01, 58: 0.01, 73: 0.01, 97: 0.01, 99: 0.01, 5: 0.09, 65: 0.02, 67: 0.45000000000000023, 25: 0.03, 2: 0.02, 8: 0.13999999999999999, 28: 0.02, 90: 0.11999999999999998, 95: 0.03}}]
    #with 8 clusters
	#L=[[[10, 29, 39, 9, 48, 71, 66, 24], [29, 83, 10, 39, 9, 48, 71, 66, 24], [5, 6, 7, 18, 21, 32, 33, 39, 54, 63, 79, 84, 93, 97, 10, 29, 9, 48, 71, 66, 24], [9, 40, 73, 10, 29, 39, 48, 71, 66, 24], [2, 27, 47, 48, 10, 29, 39, 9, 71, 66, 24], [34, 64, 71, 78, 10, 29, 39, 9, 48, 66, 24], [0, 1, 3, 4, 8, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 23, 25, 26, 28, 30, 31, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 65, 66, 67, 68, 69, 70, 72, 74, 75, 76, 77, 80, 81, 82, 85, 86, 87, 88, 89, 91, 92, 94, 96, 98, 99, 10, 29, 39, 9, 48, 71, 24], [24, 90, 95, 10, 29, 39, 9, 48, 71, 66]], {0: {11: 0.01, 30: 0.02, 40: 0.13999999999999999, 10: 0.03, 49: 0.04, 72: 0.04, 67: 0.6900000000000004, 25: 0.03}, 1: {30: 0.01, 84: 0.01, 11: 0.01, 40: 0.13999999999999999, 10: 0.03, 49: 0.04, 72: 0.04, 67: 0.6900000000000004, 25: 0.03}, 2: {6: 0.01, 7: 0.01, 8: 0.01, 19: 0.01, 22: 0.01, 33: 0.01, 34: 0.01, 40: 0.01, 55: 0.01, 64: 0.01, 80: 0.01, 85: 0.01, 94: 0.01, 98: 0.01, 11: 0.01, 30: 0.02, 10: 0.03, 49: 0.04, 72: 0.04, 67: 0.6900000000000004, 25: 0.03}, 3: {10: 0.01, 41: 0.01, 74: 0.01, 11: 0.01, 30: 0.02, 40: 0.13999999999999999, 49: 0.04, 72: 0.04, 67: 0.6900000000000004, 25: 0.03}, 4: {3: 0.01, 28: 0.01, 48: 0.01, 49: 0.01, 11: 0.01, 30: 0.02, 40: 0.13999999999999999, 10: 0.03, 72: 0.04, 67: 0.6900000000000004, 25: 0.03}, 5: {35: 0.01, 65: 0.01, 72: 0.01, 79: 0.01, 11: 0.01, 30: 0.02, 40: 0.13999999999999999, 10: 0.03, 49: 0.04, 67: 0.6900000000000004, 25: 0.03}, 6: {1: 0.01, 2: 0.01, 4: 0.01, 5: 0.01, 9: 0.01, 12: 0.01, 13: 0.01, 14: 0.01, 15: 0.01, 16: 0.01, 17: 0.01, 18: 0.01, 20: 0.01, 21: 0.01, 23: 0.01, 24: 0.01, 26: 0.01, 27: 0.01, 29: 0.01, 31: 0.01, 32: 0.01, 36: 0.01, 37: 0.01, 38: 0.01, 39: 0.01, 42: 0.01, 43: 0.01, 44: 0.01, 45: 0.01, 46: 0.01, 47: 0.01, 50: 0.01, 51: 0.01, 52: 0.01, 53: 0.01, 54: 0.01, 56: 0.01, 57: 0.01, 58: 0.01, 59: 0.01, 60: 0.01, 61: 0.01, 62: 0.01, 63: 0.01, 66: 0.01, 67: 0.01, 68: 0.01, 69: 0.01, 70: 0.01, 71: 0.01, 73: 0.01, 75: 0.01, 76: 0.01, 77: 0.01, 78: 0.01, 81: 0.01, 82: 0.01, 83: 0.01, 86: 0.01, 87: 0.01, 88: 0.01, 89: 0.01, 90: 0.01, 92: 0.01, 93: 0.01, 95: 0.01, 97: 0.01, 99: 0.01, 100: 0.01, 11: 0.01, 30: 0.02, 40: 0.13999999999999999, 10: 0.03, 49: 0.04, 72: 0.04, 25: 0.03}, 7: {25: 0.01, 91: 0.01, 96: 0.01, 11: 0.01, 30: 0.02, 40: 0.13999999999999999, 10: 0.03, 49: 0.04, 72: 0.04, 67: 0.6900000000000004}}]
	#with 5 clusters
    #L=[[[0,6,8,9,16,19,24,29,34,35,46,49,50,53,57,60,72,76,78,90,86,25,66,96],[2,3,4,5,7,12,15,18,21,31,32,33,37,38,39,44,47,48,56,58,63,67,68,75,77,79,82,84,86,87,89,91,93,95,8,25,66,96],[25, 36, 41, 62, 94, 8, 86, 66, 96],[1,10,11,13,14,17,20,23,26,27,28,30,40,43,45,51,52,54,55,59,61,64,65,66,69,70,71,73,74,80,81,85,88,92,97,8,86,25,96],[22, 42, 83, 96, 98, 99, 8, 86, 25, 66]],{0: {1: 0.01,7: 0.01,9: 0.01,10: 0.01,17: 0.01,20: 0.01,25: 0.01,30: 0.01,35: 0.01,36: 0.01,47: 0.01,50: 0.01,51: 0.01,54: 0.01,58: 0.01,61: 0.01,73: 0.01,77: 0.01,79: 0.01,91: 0.01,87: 0.34000000000000014,26: 0.05,67: 0.35000000000000014,97: 0.060000000000000005},1: {3: 0.01,4: 0.01,5: 0.01,6: 0.01,8: 0.01,13: 0.01,16: 0.01,19: 0.01,22: 0.01,32: 0.01,33: 0.01,34: 0.01,38: 0.01,39: 0.01,40: 0.01,45: 0.01,48: 0.01,49: 0.01,57: 0.01,59: 0.01,64: 0.01,68: 0.01,69: 0.01,76: 0.01,78: 0.01,80: 0.01,83: 0.01,85: 0.01,87: 0.01,88: 0.01,90: 0.01,92: 0.01,94: 0.01,96: 0.01,9: 0.20000000000000004,26: 0.05,67: 0.35000000000000014,97: 0.060000000000000005},2: {26: 0.01,37: 0.01,42: 0.01,63: 0.01,95: 0.01,9: 0.20000000000000004,87: 0.34000000000000014,67: 0.35000000000000014,97: 0.060000000000000005},3: {2: 0.01,11: 0.01,12: 0.01,14: 0.01,15: 0.01,18: 0.01,21: 0.01,24: 0.01,27: 0.01,28: 0.01,29: 0.01,31: 0.01,41: 0.01,44: 0.01,46: 0.01,52: 0.01,53: 0.01,55: 0.01,56: 0.01,60: 0.01,62: 0.01,65: 0.01,66: 0.01,67: 0.01,70: 0.01,71: 0.01,72: 0.01,74: 0.01,75: 0.01,81: 0.01,82: 0.01,86: 0.01,89: 0.01,93: 0.01,98: 0.01,9: 0.20000000000000004,87: 0.34000000000000014,26: 0.05,97: 0.060000000000000005},4: {23: 0.01,43: 0.01,84: 0.01,97: 0.01,99: 0.01,100: 0.01,9: 0.20000000000000004,87: 0.34000000000000014,26: 0.05,67: 0.35000000000000014}}]
    #without clusters
    L=[[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]], {0: {1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.01, 6: 0.01, 7: 0.01, 8: 0.01, 9: 0.01, 10: 0.01, 11: 0.01, 12: 0.01, 13: 0.01, 14: 0.01, 15: 0.01, 16: 0.01, 17: 0.01, 18: 0.01, 19: 0.01, 20: 0.01, 21: 0.01, 22: 0.01, 23: 0.01, 24: 0.01, 25: 0.01, 26: 0.01, 27: 0.01, 28: 0.01, 29: 0.01, 30: 0.01, 31: 0.01, 32: 0.01, 33: 0.01, 34: 0.01, 35: 0.01, 36: 0.01, 37: 0.01, 38: 0.01, 39: 0.01, 40: 0.01, 41: 0.01, 42: 0.01, 43: 0.01, 44: 0.01, 45: 0.01, 46: 0.01, 47: 0.01, 48: 0.01, 49: 0.01, 50: 0.01, 51: 0.01, 52: 0.01, 53: 0.01, 54: 0.01, 55: 0.01, 56: 0.01, 57: 0.01, 58: 0.01, 59: 0.01, 60: 0.01, 61: 0.01, 62: 0.01, 63: 0.01, 64: 0.01, 65: 0.01, 66: 0.01, 67: 0.01, 68: 0.01, 69: 0.01, 70: 0.01, 71: 0.01, 72: 0.01, 73: 0.01, 74: 0.01, 75: 0.01, 76: 0.01, 77: 0.01, 78: 0.01, 79: 0.01, 80: 0.01, 81: 0.01, 82: 0.01, 83: 0.01, 84: 0.01, 85: 0.01, 86: 0.01, 87: 0.01, 88: 0.01, 89: 0.01, 90: 0.01, 91: 0.01, 92: 0.01, 93: 0.01, 94: 0.01, 95: 0.01, 96: 0.01, 97: 0.01, 98: 0.01, 99: 0.01, 100: 0.01}}]
    #for 50 scenar
    #L=[[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]], {0: {1: 0.02, 2: 0.02, 3: 0.02, 4: 0.02, 5: 0.02, 6: 0.02, 7: 0.02, 8: 0.02, 9: 0.02, 10: 0.02, 11: 0.02, 12: 0.02, 13: 0.02, 14: 0.02, 15: 0.02, 16: 0.02, 17: 0.02, 18: 0.02, 19: 0.02, 20: 0.02, 21: 0.02, 22: 0.02, 23: 0.02, 24: 0.02, 25: 0.02, 26: 0.02, 27: 0.02, 28: 0.02, 29: 0.02, 30: 0.02, 31: 0.02, 32: 0.02, 33: 0.02, 34: 0.02, 35: 0.02, 36: 0.02, 37: 0.02, 38: 0.02, 39: 0.02, 40: 0.02, 41: 0.02, 42: 0.02, 43: 0.02, 44: 0.02, 45: 0.02, 46: 0.02, 47: 0.02, 48: 0.02, 49: 0.02,50:0.02}}]
    #with 50 scenarios and 10 clusters
    #L=[[[1, 13, 20, 3, 12, 22, 25, 37, 18, 42, 45, 2], [3, 4, 5, 7, 10, 16, 17, 26, 33, 38, 39, 44, 1, 12, 22, 25, 37, 18, 42, 45, 2], [12, 1, 3, 22, 25, 37, 18, 42, 45, 2], [22, 29, 1, 3, 12, 25, 37, 18, 42, 45, 2], [11, 25, 28, 36, 41, 1, 3, 12, 22, 37, 18, 42, 45, 2], [31, 37, 47, 1, 3, 12, 22, 25, 18, 42, 45, 2], [18, 40, 1, 3, 12, 22, 25, 37, 42, 45, 2], [6, 30, 34, 42, 46, 1, 3, 12, 22, 25, 37, 18, 45, 2], [0, 8, 9, 15, 19, 24, 32, 35, 43, 45, 49, 1, 3, 12, 22, 25, 37, 18, 42, 2], [2, 14, 21, 23, 27, 48, 1, 3, 12, 22, 25, 37, 18, 42, 45]], {0: {2: 0.02, 14: 0.02, 21: 0.02, 4: 0.23999999999999996, 13: 0.02, 23: 0.04, 26: 0.1, 38: 0.06, 19: 0.04, 43: 0.1, 46: 0.21999999999999997, 3: 0.12000000000000001}, 1: {4: 0.02, 5: 0.02, 6: 0.02, 8: 0.02, 11: 0.02, 17: 0.02, 18: 0.02, 27: 0.02, 34: 0.02, 39: 0.02, 40: 0.02, 45: 0.02, 2: 0.06, 13: 0.02, 23: 0.04, 26: 0.1, 38: 0.06, 19: 0.04, 43: 0.1, 46: 0.21999999999999997, 3: 0.12000000000000001}, 2: {13: 0.02, 2: 0.06, 4: 0.23999999999999996, 23: 0.04, 26: 0.1, 38: 0.06, 19: 0.04, 43: 0.1, 46: 0.21999999999999997, 3: 0.12000000000000001}, 3: {23: 0.02, 30: 0.02, 2: 0.06, 4: 0.23999999999999996, 13: 0.02, 26: 0.1, 38: 0.06, 19: 0.04, 43: 0.1, 46: 0.21999999999999997, 3: 0.12000000000000001}, 4: {12: 0.02, 26: 0.02, 29: 0.02, 37: 0.02, 42: 0.02, 2: 0.06, 4: 0.23999999999999996, 13: 0.02, 23: 0.04, 38: 0.06, 19: 0.04, 43: 0.1, 46: 0.21999999999999997, 3: 0.12000000000000001}, 5: {32: 0.02, 38: 0.02, 48: 0.02, 2: 0.06, 4: 0.23999999999999996, 13: 0.02, 23: 0.04, 26: 0.1, 19: 0.04, 43: 0.1, 46: 0.21999999999999997, 3: 0.12000000000000001}, 6: {19: 0.02, 41: 0.02, 2: 0.06, 4: 0.23999999999999996, 13: 0.02, 23: 0.04, 26: 0.1, 38: 0.06, 43: 0.1, 46: 0.21999999999999997, 3: 0.12000000000000001}, 7: {7: 0.02, 31: 0.02, 35: 0.02, 43: 0.02, 47: 0.02, 2: 0.06, 4: 0.23999999999999996, 13: 0.02, 23: 0.04, 26: 0.1, 38: 0.06, 19: 0.04, 46: 0.21999999999999997, 3: 0.12000000000000001}, 8: {1: 0.02, 9: 0.02, 10: 0.02, 16: 0.02, 20: 0.02, 25: 0.02, 33: 0.02, 36: 0.02, 44: 0.02, 46: 0.02, 50: 0.02, 2: 0.06, 4: 0.23999999999999996, 13: 0.02, 23: 0.04, 26: 0.1, 38: 0.06, 19: 0.04, 43: 0.1, 3: 0.12000000000000001}, 9: {3: 0.02, 15: 0.02, 22: 0.02, 24: 0.02, 28: 0.02, 49: 0.02, 2: 0.06, 4: 0.23999999999999996, 13: 0.02, 23: 0.04, 26: 0.1, 38: 0.06, 19: 0.04, 43: 0.1, 46: 0.21999999999999997}}]
	#    parser.add_argument("L", help = "scenarios and probabilities", type = list)
    parser.add_argument("i", help = "index of cluster", type = int) #define the index which is given as an argument in the server
#    parser.add_argument("CS", help = "number of CS", type = int)
    args = parser.parse_args()
#    print(args.N, args.CS, args.eta, args.renew, args.method, args.Time, args.StartTime)
#    main(args.N, args.CS, args.eta, args.renew, args.method, args.Time, args.StartTime)
    H= SUCperCluster(L,args.i)
    print(H)
