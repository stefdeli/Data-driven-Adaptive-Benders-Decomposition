# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 19:57:45 2016

@author: stde
"""
import gurobipy as gb
import defaults



#==============================================================================
# Day-ahead market variables
#==============================================================================

def build_variables_DA(self):      

    m = self.model
    var = self.variables
    generators = self.data.generators
    windfarms = self.data.windfarms 
#    nodes = self.data.nodes
    SchedulingHorizon=defaults.SchedulingHorizon
    NScen=defaults.NScen
    multicut=defaults.multicut
    nodenb=defaults.nodenb      
       
                
   # Dispatchable generators
    var.Pgen = {}
    for i in generators:
        var.Pgen[i]=[0]*SchedulingHorizon
        for j in range(SchedulingHorizon):
            var.Pgen[i][j] = m.addVar(lb=0.0, name = 'Pgen({0})_time({1})'.format(i,j))
    #add param for horizon 
              
   # Non-Dispatchable generators (Wind)
    var.WindDA = {}
    for j in windfarms:
        var.WindDA[j]=[0]*SchedulingHorizon
        for k in range(SchedulingHorizon):
            var.WindDA[j][k] = m.addVar(lb=0.0, name = 'WindDA({0})_time({1})'.format(j,k))
        
   # Positive Reserve Capacity procurement of conventionals units 
    var.Resp = {}
    for k in generators:
         var.Resp[k]=[0]*SchedulingHorizon
         for l in range(SchedulingHorizon):
             var.Resp[k][l]=m.addVar(lb=0.0, name = 'Resp({0})_time({1})'.format(k,l))
        
   # Negative Reserve Capacity procurement of conventionals units 
    var.Resn = {}
    for l in generators:
        var.Resn[l]=[0]*SchedulingHorizon
        for n in range(SchedulingHorizon):
            var.Resn[l][n]=m.addVar(lb=0.0, name = 'Resn({0})_time({1})'.format(l,n)) 
        
        
   #Unit status (online/offline)
    var.Stat = {}
    for n in generators:
        var.Stat[n]=[0]*SchedulingHorizon
        for p in range(SchedulingHorizon):
            var.Stat[n][p]=m.addVar(lb=0.0,vtype=gb.GRB.BINARY, name = 'Stat({0})_time({1})'.format(n,p))
        
   #starting up of unit (at a certain instant like in other variables definitions)
    var.Stup = {}
    for p in generators:
        var.Stup[p]=[0]*SchedulingHorizon
        for q in range(SchedulingHorizon):
            var.Stup[p][q]=m.addVar(lb=0.0,vtype=gb.GRB.BINARY, name = 'Stup({0})_time({1})'.format(p,q))
   
   #Shutting down of unit (at a certain instant like in other variables definitions)
    var.Shdwn = {}
    for q in generators:
        var.Shdwn[q]=[0]*SchedulingHorizon
        for r in range(SchedulingHorizon):
            var.Shdwn[q][r]=m.addVar(lb=0.0,vtype=gb.GRB.BINARY, name = 'Shdwn({0})_time({1})'.format(q,r))
        
    #Variable to indicate single or multicut
    var.theta={}
    if multicut==1:
        for i in self.scenar:
            var.theta[i]=m.addVar(lb=-2e5)
    else: var.theta=m.addVar(lb=-2e5)
    
    #Node angle for Day ahead
    var.deltaDA={}
    for i in self.data.index:
        var.deltaDA[i]=[0]*SchedulingHorizon
        for t in range(SchedulingHorizon):
            var.deltaDA[i][t]=m.addVar(lb=-gb.GRB.INFINITY,ub=gb.GRB.INFINITY,name='node_angle_DA_{0}'.format(i))
        
    #flow for each line day ahead
    var.lineflowDA={}
    for l in self.data.linesindex:
        var.lineflowDA[l]=[0]*SchedulingHorizon
        for t in range(SchedulingHorizon):
            var.lineflowDA[l][t]=m.addVar(lb=-gb.GRB.INFINITY,ub=gb.GRB.INFINITY,name='lineflow_DA_{0}'.format(l))
    
######## For Real time balancing ######### 
def build_variables_sb(self,scenar):
    m = self.model
    var = self.variables
    generators = self.data.generators
    windfarms = self.data.windfarms 
#    nodes = self.data.nodes
    SchedulingHorizon=defaults.SchedulingHorizon
    NScen=defaults.NScen
#    multicut=defaults.multicut
    nodenb=defaults.nodenb   
    
     # Non-Dispatchable generators (Wind)
    var.WindDA = {}
    for j in windfarms:
        var.WindDA[j]=[0]*SchedulingHorizon
        for k in range(SchedulingHorizon):
            var.WindDA[j][k] = m.addVar(lb=0.0, name = 'WindDA({0})_time({1})'.format(j,k))
        
   # Positive Reserve Capacity procurement of conventionals units 
    var.Resp = {}
    for k in generators:
         var.Resp[k]=[0]*SchedulingHorizon
         for l in range(SchedulingHorizon):
             var.Resp[k][l]=m.addVar(lb=0.0, name = 'Resp({0})_time({1})'.format(k,l))
        
   # Negative Reserve Capacity procurement of conventionals units 
    var.Resn = {}
    for l in generators:
        var.Resn[l]=[0]*SchedulingHorizon
        for n in range(SchedulingHorizon):
            var.Resn[l][n]=m.addVar(lb=0.0, name = 'Resn({0})_time({1})'.format(l,n)) 
    
    # Upward regulation
    var.powerUp={}
    for i in generators:
        var.powerUp[i]=[0]*SchedulingHorizon
        for j in range(SchedulingHorizon):
#            var.powerUp[i][j]={}
#            for n in scenar:    
#                var.powerUp[i][j][n]=m.addVar(lb=0.0,name='powerupgene({0})_time({1})_scenar({2})'.format(i,j,n))
            #for parallel MP    
            var.powerUp[i][j]=m.addVar(lb=0.0,name='powerupgene({0})_time({1})_'.format(i,j))
    
        
   # Downward regulation
    var.powerDn={}
    for i in generators:
        var.powerDn[i]=[0]*SchedulingHorizon
        for j in range(SchedulingHorizon):
#            var.powerDn[i][j]={}
#            for n in scenar:    
#                var.powerDn[i][j][n]=m.addVar(lb=0.0,name='powerdownngene({0})_time({1})_scenar({2})'.format(i,j,n))
            #for parallel MP
            var.powerDn[i][j]=m.addVar(lb=0.0,name='powerdownngene({0})_time({1})'.format(i,j))
            
    # Wind spillage
    var.wspill={}
    for i in windfarms:
        var.wspill[i]=[0]*SchedulingHorizon
        for j in range(SchedulingHorizon):
#            var.wspill[i][j]={}
#            for n in scenar:
#                var.wspill[i][j][n]=m.addVar(lb=0.0,name='wspill windfarm({0})_time({1})_scenar({2})'.format(i,j,n))
            #for parallel MP
            var.wspill[i][j]=m.addVar(lb=0.0,name='wspill windfarm({0})_time({1})'.format(i,j))
            
   
    #flow for each line day ahead
    var.lineflowDA={}
    for l in self.data.linesindex:
        var.lineflowDA[l]=[0]*SchedulingHorizon
        for t in range(SchedulingHorizon):
            var.lineflowDA[l][t]=m.addVar(lb=-gb.GRB.INFINITY,ub=gb.GRB.INFINITY,name='lineflow_DA_{0}'.format(l))

    var.lshed={}
    for i in self.data.index:
        var.lshed[i]=[0]*SchedulingHorizon
        for j in range(SchedulingHorizon):
#            var.lshed[i][j]={}
#            for n in scenar:
#                var.lshed[i][j][n]=m.addVar(lb=0.0, name='load_shedding_node({0})_time({1})_scenar({2})'.format(i,j,n))
            #for parallel MP
            var.lshed[i][j]=m.addVar(lb=0.0, name='load_shedding_node({0})_time({1})'.format(i,j))
                    
    #Nodal phase angle subpb(real time additional)
    var.deltaSB={}
    for i in self.data.index:
        var.deltaSB[i]=[0]*SchedulingHorizon
        for t in range(SchedulingHorizon):
#            var.deltaSB[i][t]={}
#            for n in scenar:
#                var.deltaSB[i][t][n]=m.addVar(lb=-gb.GRB.INFINITY,ub=gb.GRB.INFINITY,name='node_angle_SB_{0}'.format(i))
            #for parallel MP
            var.deltaSB[i][t]=m.addVar(lb=-gb.GRB.INFINITY,ub=gb.GRB.INFINITY,name='node_angle_SB_{0}'.format(i))
    
    #flow for each line subpb
    var.lineflowSB={}
    for l in self.data.linesindex:
        var.lineflowSB[l]=[0]*SchedulingHorizon
        for t in range(SchedulingHorizon):
#            var.lineflowSB[l][t]={}
#            for n in scenar:
#                var.lineflowSB[l][t][n]=m.addVar(lb=-gb.GRB.INFINITY,ub=gb.GRB.INFINITY,name='lineflow_DA_{0}'.format(l))
            #for parallel MP
            var.lineflowSB[l][t]=m.addVar(lb=-gb.GRB.INFINITY,ub=gb.GRB.INFINITY,name='lineflow_DA_{0}'.format(l))




