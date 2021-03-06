# -*- coding: utf-8 -*-
'''
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
'''


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
        
    

    
