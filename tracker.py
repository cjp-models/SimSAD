import pandas as pd
import numpy as np
import os
from itertools import product
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'SimSAD/data')
pd.options.mode.chained_assignment = None

class tracker_entry:
    def __init__(self,entry_name, class_member, rowvars,colvars,aggfunc,start_yr,stop_yr):
        self.entry_name = entry_name
        self.class_member = class_member
        self.rowvars = rowvars
        self.colvars = colvars
        self.aggfunc = aggfunc
        self.start_yr = start_yr 
        self.stop_yr = stop_yr
        self.table = []
        return 
        
class tracker:
    def __init__(self):
        self.registry = []
        return 
    def add_entry(self,entry_name, class_member, rowvars,colvars,aggfunc,start_yr,stop_yr):
        entry = tracker_entry(entry_name,class_member,rowvars,colvars,aggfunc,start_yr,stop_yr)
        self.registry.append(entry)
        return 
    def log(self,p, yr):
        for k in self.registry:
            c = getattr(p,k.class_member)
            table = c.collapse(rowvars=k.rowvars,colvars=k.colvars).stack()
            if k.colvars==[]:
                table = table.droplevel(1)
            if yr==k.start_yr:
                k.table = pd.DataFrame(index=table.index,columns=np.arange(k.start_yr,k.stop_yr+1))
            k.table[yr] = table
        return 
    def save(self):
        for k in self.registry:
            k.table.to_excel(k.entry_name+'.xlsx')
        return 
            
        
