# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 13:37:04 2016

Library of Objective Functions

@author: stde
"""
import gurobipy as gb
from defaults import SchedulingHorizon,nodenb,VOLL,multicut,NScen


 # Objective Function - Day Ahead Market        
def build_objective_masterpb(self):        
    
    generators = self.data.generators
#    gendata = self.data.generatorinfo   
    generatorsdata=self.data.generatordata
    m = self.model
    
    if multicut==1:
         m.setObjective(
        (gb.quicksum((generatorsdata.LinCost[i]*self.variables.Pgen[i][t]+generatorsdata.StrtUpCost[i]*self.variables.Stup[i][t]+0.2*generatorsdata.LinCost[i]*self.variables.Resp[i][t]+0.2*generatorsdata.LinCost[i]*self.variables.Resn[i][t]) for i in generators for t in range(SchedulingHorizon))
        +gb.quicksum(self.variables.theta[sc]*self.proba[sc+1] for sc in self.scenar)),        
        gb.GRB.MINIMIZE)
    else:
         m.setObjective(
        (gb.quicksum(generatorsdata.LinCost[i]*self.variables.Pgen[i][t]+generatorsdata.StrtUpCost[i]*self.variables.Stup[i][t]+0.2*generatorsdata.LinCost[i]*self.variables.Resp[i][t]+0.2*generatorsdata.LinCost[i]*self.variables.Resn[i][t] for i in generators for t in range(SchedulingHorizon))
        +self.variables.theta),        
        gb.GRB.MINIMIZE)
        
    

def build_objective_subpb(self,w):        
    
    generators = self.data.generators
#    gendata = self.MP.data.generatorinfo
    generatorsdata=self.data.generatordata

    m = self.model
           
    m.setObjective(
        gb.quicksum((generatorsdata.UpRegCost[i]*self.variables.powerUp[i][t]-generatorsdata.DnRegCost[i]*self.variables.powerDn[i][t])for i in generators for t in range(SchedulingHorizon))+gb.quicksum(VOLL*self.variables.lshed[n][t]  for t in range(SchedulingHorizon) for n in self.data.index),        
        gb.GRB.MINIMIZE)
    