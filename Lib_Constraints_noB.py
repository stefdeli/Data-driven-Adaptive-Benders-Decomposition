# -*- coding: utf-8 -*-
"""
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

"""
import gurobipy as gb
import defaults
SchedulingHorizon=defaults.SchedulingHorizon
    
#===========================================================================
# Day-ahead market constraints
#==============================================================================
def build_masterpb_constr(self):
    var=self.variables
    nb=0
    masterpb={}
#initialisation of the status of the units according to up/down time and past situation
    self.LUP={}
    self.LDN={}
    for i in self.data.generators:
        self.T=self.data.generatordata.Tini[i]
        if self.T>=0:
            if self.T>self.data.generatorinfo.MinUpTime[i]:
                self.LUP[i]=0
            else:
                self.LUP[i]=self.data.generatorinfo.MinUpTime[i]-self.T
            self.LDN[i]=0
        else:
            self.LUP[i]=0
            if -self.T>=self.data.generatorinfo.MinDnTime[i]:
                self.LDN[i]=0
            else:
                self.LDN[i]=self.data.generatorinfo.MinDnTime[i]+self.T
#constraint on initial state
    for i in self.data.generators:
        if (self.LUP[i]+self.LDN[i])!=0:
            if (self.LUP[i]+self.LDN[i])<defaults.SchedulingHorizon :
                for t in range(0,self.LUP[i]+self.LDN[i]):
                    masterpb[nb]=self.model.addConstr(var.Stat[i][t],gb.GRB.EQUAL,self.data.generatordata.Statini[i])
                    nb+=1
            else:
                for t in range(0,SchedulingHorizon):
                    masterpb[nb]=self.model.addConstr(var.Stat[i][t],gb.GRB.EQUAL,self.data.generatordata.Statini[i])
                    nb+=1
##minimum up and down time
    for i in self.data.generators:
        UT=self.data.generatorinfo.MinUpTime[i]
        DT=self.data.generatorinfo.MinDnTime[i]
        if (self.LUP[i]+self.LDN[i])<SchedulingHorizon:
            for t in range(self.LUP[i]+self.LDN[i],SchedulingHorizon):
                if SchedulingHorizon>UT:
                    #min up time
                    masterpb[nb]=self.model.addConstr(var.Stat[i][t],gb.GRB.GREATER_EQUAL,gb.quicksum(var.Stup[i][tau] for tau in range(t-UT+1,t)))
                    nb+=1
                else:
                    masterpb[nb]=self.model.addConstr(var.Stat[i][t],gb.GRB.GREATER_EQUAL,gb.quicksum(var.Stup[i][tau] for tau in range(t-self.LUP[i],t)))
                    nb+=1  
                #min down time
                if SchedulingHorizon>DT:
                    masterpb[nb]=self.model.addConstr(1-var.Stat[i][t],gb.GRB.GREATER_EQUAL,gb.quicksum(var.Shdwn[i][tau] for tau in range(t-DT,t)))
                    nb+=1
                else:
                     masterpb[nb]=self.model.addConstr(1-var.Stat[i][t],gb.GRB.GREATER_EQUAL,gb.quicksum(var.Shdwn[i][tau] for tau in range(t-self.LDN[i],t)))
                     nb+=1
#initialisation of the power output
        masterpb[nb]=self.model.addConstr(var.Pgen[i][0]+var.Resp[i][0],gb.GRB.LESS_EQUAL,(self.data.generatordata.Pini[i]+self.data.generatorinfo.RampUp[i])*var.Stat[i][0])
        nb+=1
        masterpb[nb]=self.model.addConstr(var.Pgen[i][0]-var.Resn[i][0],gb.GRB.GREATER_EQUAL,(self.data.generatordata.Pini[i]-self.data.generatorinfo.RampDn[i])*var.Stat[i][0])
        nb+=1 
#(1c) start-up/shut-down
    for i in self.data.generators:   
        masterpb[nb]=self.model.addConstr(var.Stup[i][0]-var.Shdwn[i][0],gb.GRB.EQUAL,var.Stat[i][0]-self.data.generatordata.Statini[i])
        nb+=1
        for t in range(1,SchedulingHorizon):
            masterpb[nb]=self.model.addConstr(var.Stup[i][t]-var.Shdwn[i][t],gb.GRB.EQUAL,var.Stat[i][t]-var.Stat[i][t-1])
            nb+=1
##(1d) restrict simultaneous start-up and shut-down
        for t in range(SchedulingHorizon):    
            masterpb[nb]=self.model.addConstr(var.Stup[i][t]+var.Shdwn[i][t],gb.GRB.LESS_EQUAL,1)
            nb+=1
#(1e) unit turned on at the beginning of start up
            masterpb[nb]=self.model.addConstr(var.Stup[i][t],gb.GRB.LESS_EQUAL,var.Stat[i][t])
            nb+=1
#(1f) unit on at t if shutting down at t+1
        for t in range(1,SchedulingHorizon):
            masterpb[nb]=self.model.addConstr(var.Shdwn[i][t],gb.GRB.LESS_EQUAL,var.Stat[i][t-1])
            nb+=1
# Ramping Constraints (constraint on the power difference between two periods)
        for t in range(1,SchedulingHorizon):
            masterpb[nb]=self.model.addConstr(var.Pgen[i][t]+var.Resp[i][t],gb.GRB.LESS_EQUAL,var.Pgen[i][t-1]+var.Resp[i][t-1]+self.data.generatorinfo.RampUp[i]*(var.Stat[i][t-1]+var.Stup[i][t]))
            nb+=1
            masterpb[nb]=self.model.addConstr(var.Pgen[i][t-1]-var.Resn[i][t-1],gb.GRB.LESS_EQUAL,var.Pgen[i][t]-var.Resn[i][t]+self.data.generatorinfo.RampDn[i]*(var.Stat[i][t]+var.Shdwn[i][t]))
            nb+=1
#(1g) units can't produce more than capacity and must respect reserve
        for t in range (0,SchedulingHorizon-1):
            masterpb[nb]=self.model.addConstr(var.Pgen[i][t]+var.Resp[i][t],gb.GRB.LESS_EQUAL,self.data.generatorinfo.Pmax[i]*(var.Stat[i][t]-var.Shdwn[i][t+1])+self.data.generatorinfo.RampDn[i]*var.Shdwn[i][t+1])
            nb+=1
##(1h) units have to respect lower capacity limit and
        for t in range (0,SchedulingHorizon):
            masterpb[nb]=self.model.addConstr(var.Pgen[i][t]-var.Resn[i][t],gb.GRB.GREATER_EQUAL,self.data.generatorinfo.Pmin[i]*var.Stat[i][t])  
#            masterpb[nb]=self.model.addConstr(var.Pgen[i][t]-var.Resn[i][t],gb.GRB.GREATER_EQUAL,0)  
            nb+=1
#(1i)+(1j) Reserve limit capacity (20% of capacity set as max value for up/downward reserve)
        #for t in range (0,SchedulingHorizon):
            masterpb[nb]=self.model.addConstr(var.Resp[i][t],gb.GRB.LESS_EQUAL,self.data.generatorinfo.UpCap[i])
            nb+=1
            masterpb[nb]=self.model.addConstr(var.Resn[i][t],gb.GRB.LESS_EQUAL,self.data.generatorinfo.DnCap[i])
            nb+=1
#initialisation of the nodal angles: slack bus is node 13
    for t in range(SchedulingHorizon):        
        masterpb[nb]=self.model.addConstr(var.deltaDA[101][t],gb.GRB.EQUAL,0)
        nb+=1
#(1k)limit on wind farm production (physical limit)
    for j in self.data.windfarms:
        for t in range(SchedulingHorizon):        
            masterpb[nb]=self.model.addConstr(var.WindDA[j][t],gb.GRB.LESS_EQUAL,self.data.windinfo.Capacity[j])
            nb+=1
#power balance for each node
    for n in self.data.index:
        for t in range(SchedulingHorizon):
            masterpb[nb]=self.model.addConstr(gb.quicksum(var.Pgen[i][t] for i in self.data.N2G[n])+gb.quicksum(var.WindDA[j][t] for j in self.data.N2W[n]),
                    gb.GRB.EQUAL,gb.quicksum(self.data.load['{0}'.format(i)][t+1] for i in self.data.N2L[n])+gb.quicksum(var.lineflowDA[l][t] for l in self.data.N2F[n]['pos'])-gb.quicksum(var.lineflowDA[l][t] for l in self.data.N2F[n]['neg']))
            nb+=1
##lineflow limit
    for t in range(SchedulingHorizon):
        for l in self.data.linesindex:
            masterpb[nb]=self.model.addConstr(var.lineflowDA[l][t],gb.GRB.GREATER_EQUAL,-(self.data.linedf.Cap[l]))
            nb+=1
            masterpb[nb]=self.model.addConstr(var.lineflowDA[l][t],gb.GRB.LESS_EQUAL,self.data.linedf.Cap[l])
            nb+=1
#compute flow from angles
            masterpb[nb]=self.model.addConstr(var.lineflowDA[l][t],gb.GRB.EQUAL,(var.deltaDA[l[0]][t]-var.deltaDA[l[1]][t])*self.data.linedf.React[l])
            nb+=1

##(1l) balance of the redispatch
#    for w in range(defaults.NScen):
    for w in range(1,defaults.NScen+1):
        for t in range(SchedulingHorizon):
            #initialisation of the nodal angles: slack bus is node 13
            masterpb[nb]=self.model.addConstr(var.deltaSB[101][t][w],gb.GRB.EQUAL,0)
            nb+=1
            for n in self.data.index:
                masterpb[nb]=self.model.addConstr(gb.quicksum((var.powerUp[i][t][w]-var.powerDn[i][t][w]) for i in self.data.N2G[n])+gb.quicksum((self.data.windscen[j]['{0}'.format(w)][t+1]*self.data.windinfo.Capacity[j]-var.WindDA[j][t]-var.wspill[j][t][w]) for j in self.data.N2W[n])+gb.quicksum(var.lshed[d][t][w] for d in self.data.N2L[n]),gb.GRB.EQUAL,
                               gb.quicksum((self.variables.lineflowSB[l][t][w]-self.variables.lineflowDA[l][t]) for l in self.data.N2F[n]['pos'])+gb.quicksum((-self.variables.lineflowSB[k][t][w]+self.variables.lineflowDA[k][t]) for k in self.data.N2F[n]['neg'] ))
                nb+=1

#(1m)&(1n) regulation smaller than reserve capacity
            for i in self.data.generators:
                masterpb[nb]=self.model.addConstr(var.powerUp[i][t][w],gb.GRB.LESS_EQUAL,var.Resp[i][t])
                nb+=1
                masterpb[nb]=self.model.addConstr(var.powerDn[i][t][w],gb.GRB.LESS_EQUAL,var.Resn[i][t])
                nb+=1
#(1o) can't shed more than the full load
    #for t in range(SchedulingHorizon):
            for n in self.data.index:
                masterpb[nb]=self.model.addConstr(var.lshed[n][t][w],gb.GRB.LESS_EQUAL,self.data.load['{0}'.format(n)][t+1])
                nb+=1
#(1p) wind spillage should be smaller than the wind production
            for j in self.data.windfarms: 
                masterpb[nb]=self.model.addConstr(var.wspill[j][t][w],gb.GRB.LESS_EQUAL,self.data.windscen[j]['{0}'.format(w)][t+1]*self.data.windinfo.Capacity[j])
                nb+=1

#lineflow limit
        for t in range(SchedulingHorizon):
            for l in self.data.linesindex:
                masterpb[nb]=self.model.addConstr(-self.data.linedf.Cap[l],gb.GRB.LESS_EQUAL,var.lineflowSB[l][t][w])
                nb+=1
                masterpb[nb]=self.model.addConstr(self.data.linedf.Cap[l],gb.GRB.GREATER_EQUAL,var.lineflowSB[l][t][w])
                nb+=1
#compute flow from angles
                masterpb[nb]=self.model.addConstr(var.lineflowSB[l][t][w],gb.GRB.EQUAL,(var.deltaSB[l[0]][t][w]-var.deltaSB[l[1]][t][w])*self.data.linedf.React[l])
                nb+=1
    return masterpb

