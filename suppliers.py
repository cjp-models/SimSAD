import pandas as pd 
import numpy as np 
from itertools import product
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'SimSAD/data')
from random import choices

class eesad: 
    def __init__(self):
        return 
    def load_registry(self):
        return
    def assign(self):
        return
    def compute_needs(self):
        return
    def compute_supply(self):
        return
    def compute_serv_rate(self):
        return



class clsc:
    def __init__(self):
        self.care_types = ['inf','avq','avd']
        self.load_params()
        return 
    def load_registry(self):
        return
    def load_params(self):
        self.pars_inf_avq_home =  pd.read_csv(os.path.join(data_dir,'prob_csss_inf_avq_home.csv'),
            delimiter=';',low_memory=False)
        self.pars_inf_avq_home.set_index(['region_id','iso_smaf','gr_age'],inplace=True)
        self.pars_inf_avq_home.sort_index(inplace=True)
        self.pars_inf_avq_home = self.pars_inf_avq_home[['_00','_01','_10','_11']]
        self.pars_inf_avq_home.columns = ['inf_avq_00','inf_avq_01','inf_avq_10','inf_avq_11']
        tots = self.pars_inf_avq_home.sum(axis=1)
        for c in self.pars_inf_avq_home.columns:
            self.pars_inf_avq_home[c] = self.pars_inf_avq_home[c]/tots
        self.pars_inf_avq_rpa =  pd.read_csv(os.path.join(data_dir,'prob_csss_inf_avq_rpa.csv'),
            delimiter=';',low_memory=False)
        self.pars_inf_avq_rpa.set_index(['region_id','iso_smaf','gr_age'],inplace=True)
        self.pars_inf_avq_rpa.sort_index(inplace=True)
        self.pars_inf_avq_rpa = self.pars_inf_avq_rpa[['_00','_01','_10','_11']]
        self.pars_inf_avq_rpa.columns = ['inf_avq_00','inf_avq_01','inf_avq_10','inf_avq_11']




        #self.pars_inf_avq_rpa =  pd.read_csv(os.path.join(data_dir,'prob_csss_inf_avq_home.csv'),
        #    delimiter=';',low_memory=False)
        #self.pars_avd_home =  pd.read_csv(os.path.join(data_dir,'prob_csss_avd_home.csv'),
        #    delimiter=';',low_memory=False)
        #self.pars_avd_rpa =  pd.read_csv(os.path.join(data_dir,'prob_csss_avd_rpa.csv'),
        #    delimiter=';',low_memory=False)
        return
    def draw(self, row):
        picks = [(False,False),(False,True),(True,False),(True,True)]
        i = choices(np.arange(4), k = 1,
                                 weights=row[['inf_avq_00','inf_avq_01','inf_avq_10','inf_avq_11']])
        row['clsc_inf_any'] = picks[i[0]][0]
        row['clsc_avq_any'] = picks[i[0]][1]
        return row
    def assign(self, users, milieu):
        users['clsc_inf_any'] = False
        users['clsc_avq_any'] = False
        merge_key = ['region_id','iso_smaf','gr_age']
        if milieu == 'home':
            pars = self.pars_inf_avq_home
        if milieu == 'rpa':
            pars = self.pars_inf_avq_rpa
        select = users.index[users.any_svc]
        work = users.loc[select, :]
        work = work.merge(pars, left_on=merge_key, right_on=merge_key, how='left')
        work = work.apply(self.draw, axis=1)
        users.loc[select, :] = work
        return users
    def compute_needs(self):
        return
    def compute_supply(self):
        return
    def compute_serv_rate(self):
        return

# autres services achetés, AVQ + soins infirmiers     
class prive:
    def __init__(self):
        return 
    



