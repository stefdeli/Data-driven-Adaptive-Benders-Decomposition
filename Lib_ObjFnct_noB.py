# -*- coding: utf-8 -*-


import gurobipy as gb
from defaults import SchedulingHorizon,nodenb,VOLL,multicut,NScen


 # Objective Function - Day Ahead Market        
def build_objective_masterpb(self):        
    
    generators = self.data.generators
#    gendata = self.data.generatorinfo   
    generatorsdata=self.data.generatordata
    m = self.model

    m.setObjective(
    (gb.quicksum(generatorsdata.LinCost[i]*self.variables.Pgen[i][t]+generatorsdata.StrtUpCost[i]*self.variables.Stup[i][t]+
                 0.2*generatorsdata.LinCost[i]*self.variables.Resp[i][t]+
                 0.2*generatorsdata.LinCost[i]*self.variables.Resn[i][t] for i in generators for t in range(SchedulingHorizon))
    +gb.quicksum(self.data.scenprob[w]*(gb.quicksum((generatorsdata.UpRegCost[i]*self.variables.powerUp[i][t][w]
    -generatorsdata.DnRegCost[i]*self.variables.powerDn[i][t][w])for i in generators for t in range(SchedulingHorizon))
    +gb.quicksum(VOLL*self.variables.lshed[n][t][w] for t in range(SchedulingHorizon) for n in self.data.index))for w in range(1,NScen+1))),     
    gb.GRB.MINIMIZE)
        
    

    