# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 19:00:51 2016

@author: stde
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
#            self.LUP[i]=min(SchedulingHorizon,(self.data.generatorinfo.MinUpTime[i]-self.T)*self.data.generatordata.Statini[i])            
            self.LDN[i]=0
        else:
            self.LUP[i]=0
#            self.LDN[i]=min(SchedulingHorizon,(self.data.generatorinfo.MinDnTime[i]+self.T)*(1-self.data.generatordata.Statini[i]))
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
#minimum up and down time
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
##(1b)power balance
#    for t in range(SchedulingHorizon):
#        masterpb[nb]=self.model.addConstr(gb.quicksum(var.Pgen[i][t] for i in self.data.generators)+gb.quicksum(var.WindDA[j][t] for j in self.data.windfarms),gb.GRB.EQUAL,self.data.load[t][0]*defaults.LoadPeak)
#        nb+=1
##initialisation of Pgen
#    for i in self.data.generators:
#        masterpb[nb]=self.model.addConstr(var.Pgen[i][0],gb.GRB.EQUAL,self.data.generatorinfo.Pini[i])
#        nb+=1
#(1c) start-up/shut-down
    for i in self.data.generators:   
        masterpb[nb]=self.model.addConstr(var.Stup[i][0]-var.Shdwn[i][0],gb.GRB.EQUAL,var.Stat[i][0]-self.data.generatordata.Statini[i])
        nb+=1
        for t in range(1,SchedulingHorizon):
            masterpb[nb]=self.model.addConstr(var.Stup[i][t]-var.Shdwn[i][t],gb.GRB.EQUAL,var.Stat[i][t]-var.Stat[i][t-1])
            nb+=1
#(1d) restrict simultaneous start-up and shut-down
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
#(1h) units have to respect lower capacity limit and
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
#lineflow limit
    for t in range(SchedulingHorizon):
        for l in self.data.linesindex:
            masterpb[nb]=self.model.addConstr(-self.data.linedf.Cap[l],gb.GRB.LESS_EQUAL,var.lineflowDA[l][t])
            nb+=1
            masterpb[nb]=self.model.addConstr(self.data.linedf.Cap[l],gb.GRB.GREATER_EQUAL,var.lineflowDA[l][t])
            nb+=1
#compute flow from angles
            masterpb[nb]=self.model.addConstr(var.lineflowDA[l][t],gb.GRB.EQUAL,(var.deltaDA[l[0]][t]-var.deltaDA[l[1]][t])*self.data.linedf.React[l])
            nb+=1

###########################################
def build_subproblem_constr(self,w,Resn,Resp,WindDA,lineflowDA):
    var=self.variables
    nb=0
    self.subproblem={}
    self.subproblem_fixedp={}
    self.subproblem_fixedn={}
    self.subproblem_fixedw={}
    self.subproblem_flow={}
#(1l) balance of the redispatch
    #for w in range(defaults.NScen):
    for t in range(SchedulingHorizon):
        for n in self.data.index:
            self.subproblem[nb]=self.model.addConstr(gb.quicksum((var.powerUp[i][t]-var.powerDn[i][t]) for i in self.data.N2G[n])+gb.quicksum((self.data.windscen[j]['{0}'.format(w+1)][t+1]*self.data.windinfo.Capacity[j]-var.WindDA[j][t]-var.wspill[j][t]) for j in self.data.N2W[n])+gb.quicksum(var.lshed[b][t] for b in self.data.N2L[n]),gb.GRB.EQUAL,
                           gb.quicksum((self.variables.lineflowSB[l][t]-self.variables.lineflowDA[l][t]) for l in self.data.N2F[n]['pos'])+gb.quicksum((-self.variables.lineflowSB[k][t]+self.variables.lineflowDA[k][t]) for k in self.data.N2F[n]['neg'] ))
#initialisation of the nodal angles: slack bus is node 13
        self.subproblem[nb]=self.model.addConstr(var.deltaSB[101][t],gb.GRB.EQUAL,0)
        nb+=1
#(1m)&(1n) regulation smaller than reserve capacity
        for i in self.data.generators:
            self.subproblem[nb]=self.model.addConstr(var.powerUp[i][t],gb.GRB.LESS_EQUAL,var.Resp[i][t])
            nb+=1
            self.subproblem[nb]=self.model.addConstr(var.powerDn[i][t],gb.GRB.LESS_EQUAL,var.Resn[i][t])
            nb+=1
#(1o) can't shed more than the full load
    #for t in range(SchedulingHorizon):
        for n in self.data.index :
            self.subproblem[nb]=self.model.addConstr(var.lshed[n][t],gb.GRB.LESS_EQUAL,self.data.load['{0}'.format(n)][t+1])
            nb+=1
#(1p) wind spillage should be smaller than the wind production
        for j in self.data.windfarms: 
            self.subproblem[nb]=self.model.addConstr(var.wspill[j][t],gb.GRB.LESS_EQUAL,self.data.windscen[j]['{0}'.format(w+1)][t+1]*self.data.windinfo.Capacity[j])
            nb+=1

#(14c-d-e) fix complicating variables of the opt sol of the master pb--In case of multiprocess
    for i in self.data.generators:
         self.subproblem_fixedp[i]={}
         self.subproblem_fixedn[i]={}
         for t in range(SchedulingHorizon):
            self.subproblem_fixedp[i][t]=self.model.addConstr(var.Resp[i][t],gb.GRB.EQUAL,Resp[i][t],name='lambda_plus({0},{1},{2})'.format(i,t,w))
            self.subproblem_fixedn[i][t]=self.model.addConstr(var.Resn[i][t],gb.GRB.EQUAL,Resn[i][t],name='lambda_moins({0},{1},{2})'.format(i,t,w))
    for j in self.data.windfarms:
        self.subproblem_fixedw[j]={}
        for t in range(SchedulingHorizon):
            self.subproblem_fixedw[j][t]=self.model.addConstr(var.WindDA[j][t],gb.GRB.EQUAL,WindDA[j][t],name='lambda_wind({0},{1},{2})'.format(j,t,w))
    for l in self.data.linesindex:
        self.subproblem_flow[l]={}
        for t in range(SchedulingHorizon):
            self.subproblem_flow[l][t]=self.model.addConstr(var.lineflowDA[l][t],gb.GRB.EQUAL,lineflowDA[l][t],name='lambda flow({0},{1},{2})'.format(l,t,w))

#lineflow limit
    for t in range(SchedulingHorizon):
        for l in self.data.linesindex:
            self.subproblem[nb]=self.model.addConstr(-self.data.linedf.Cap[l],gb.GRB.LESS_EQUAL,var.lineflowSB[l][t])
            nb+=1
            self.subproblem[nb]=self.model.addConstr(self.data.linedf.Cap[l],gb.GRB.GREATER_EQUAL,var.lineflowSB[l][t])
            nb+=1
#compute flow from angles
            self.subproblem[nb]=self.model.addConstr(var.lineflowSB[l][t],gb.GRB.EQUAL,(var.deltaSB[l[0]][t]-var.deltaSB[l[1]][t])*self.data.linedf.React[l])
            nb+=1
    return self.subproblem


def update_fixed_vars(self,MP):
    for t in range(SchedulingHorizon):
        for i in self.MP.data.generators:
            self.subproblem_fixedp[i][t].rhs=MP.variables.Resp[i][t].x
            self.subproblem_fixedn[i][t].rhs=MP.variables.Resn[i][t].x
        for j in self.MP.data.windfarms:
            self.subproblem_fixedw[j][t].rhs=MP.variables.WindDA[j][t].x
        for l in self.MP.data.linesindex:
            self.subproblem_flow[l][t].rhs=MP.variables.lineflowDA[l][t].x
#    return self.subproblem_fixedp,self.subproblem_fixedn,self.subproblem_fixedw

