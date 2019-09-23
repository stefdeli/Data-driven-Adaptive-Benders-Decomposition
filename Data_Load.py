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

#==============================================================================
#  Data Loading  
#==============================================================================

import pandas as pd
import gurobipy as gb
import defaults
from collections import defaultdict
from itertools import chain

def _load_network(self):
    self.data.index=[101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325]
#    self.data.nodedf = pd.read_csv(defaults.nodefile).set_index('ID')
    self.data.linedf = pd.read_csv(defaults.linefile).set_index(['From','To'])
    self.data.linesindex = self.data.linedf.index.tolist()
    
#Map lines with node (positive when flow leaving, negative when arriving)
    self.data.N2F={}
    for i in self.data.index:
        self.data.N2F[i]={}
        self.data.N2F[i]['pos']=[]
        self.data.N2F[i]['neg']=[]
        for j in self.data.index:
            if ((i,j) in self.data.linesindex):
                self.data.N2F[i]['pos'].append((i,j))
            elif((j,i) in self.data.linesindex):
                self.data.N2F[i]['neg'].append((j,i))
            
        

    
def _load_generator_data(self):
    self.data.generatorinfo = pd.read_csv(defaults.generatorfile, index_col=0)
    self.data.generators = self.data.generatorinfo.index.tolist()
    self.data.generatordata = pd.read_csv(defaults.generator_costandstatus,index_col=0)

#Map generators with nodes 
    self.data.N2G={}
    for i in self.data.index:
        self.data.N2G[i]=[]
        for g in self.data.generators:
            if self.data.generatorinfo.Node[g]==i:
                self.data.N2G[i].append(g)
                

            
def _load_wind_data(self):
    self.data.windinfo = pd.read_csv(defaults.windfarms_file, index_col = 0)        
    self.data.windfarms = self.data.windinfo.index.tolist()    
    
#Map windfarms with nodes
    self.data.N2W={}
    for i in self.data.index:
        self.data.N2W[i]=[]
        for w in self.data.windfarms:
            if self.data.windinfo.Node[w]==i:
                self.data.N2W[i].append(w)



    
def _load_intial_data(self):
    self.data.load = pd.read_csv(defaults.load_file,index_col=0)

    
#load different scenarios for different windfarms     
    self.data.windscen={}
    self.data.windscen[1]=pd.read_csv(defaults.WindScen_file_1,index_col=0)
    self.data.windscen[2]=pd.read_csv(defaults.WindScen_file_2,index_col=0)
    self.data.windscen[3]=pd.read_csv(defaults.WindScen_file_3,index_col=0)
    self.data.windscen[4]=pd.read_csv(defaults.WindScen_file_4,index_col=0)
    self.data.windscen[5]=pd.read_csv(defaults.WindScen_file_5,index_col=0)
    self.data.windscen[6]=pd.read_csv(defaults.WindScen_file_6,index_col=0)
    self.data.windscen[7]=pd.read_csv(defaults.WindScen_file_7,index_col=0)
    self.data.windscen[8]=pd.read_csv(defaults.WindScen_file_8,index_col=0)
    self.data.windscen[9]=pd.read_csv(defaults.WindScen_file_9,index_col=0)
    self.data.windscen[10]=pd.read_csv(defaults.WindScen_file_10,index_col=0)
    self.data.windscen[11]=pd.read_csv(defaults.WindScen_file_11,index_col=0)
    self.data.windscen[12]=pd.read_csv(defaults.WindScen_file_12,index_col=0)
    self.data.windscen[13]=pd.read_csv(defaults.WindScen_file_13,index_col=0)
    self.data.windscen[14]=pd.read_csv(defaults.WindScen_file_14,index_col=0)
    self.data.windscen[15]=pd.read_csv(defaults.WindScen_file_15,index_col=0)

    self.data.scenarios = self.data.windscen[1].columns.tolist()
#Map loads to nodes
    self.data.N2L={}
    for n in self.data.index:
        self.data.N2L[n]=[]
        if '{0}'.format(n) in self.data.load.columns.tolist():
            self.data.N2L[n].append(n)
            
#        for i in self.data.demand.index.tolist():
#            if self.data.demand.Node[i]==n:
#                self.data.N2L[n].append(i)
#                
                
    # For equiprobable scenarios
    self.data.scenprob = {s: 1.0/defaults.NScen for s in range(1,defaults.NScen+1)}

        
