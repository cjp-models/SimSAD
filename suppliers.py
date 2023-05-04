import pandas as pd 
import numpy as np 
from itertools import product
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'SimSAD/data')
from numba import njit, float64, int64, boolean, prange
from numba.types import Tuple

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

        self.pars_avd_home =  pd.read_csv(os.path.join(data_dir,'prob_csss_avd_home.csv'),
            delimiter=';',low_memory=False)
        self.pars_avd_home.columns = ['region_id','iso_smaf','gr_age','_10','_01','_11','_00', 'prob_avd']
        self.pars_avd_home = self.pars_avd_home[['region_id','iso_smaf','gr_age','_00','_01','_10','_11','prob_avd']]
        self.pars_avd_home['choice'] = 0
        self.pars_avd_home['choice'] = np.where(self.pars_avd_home['_00']==1,0,self.pars_avd_home['choice'])
        self.pars_avd_home['choice'] = np.where(self.pars_avd_home['_01']==1,1,self.pars_avd_home['choice'])
        self.pars_avd_home['choice'] = np.where(self.pars_avd_home['_10']==1,2,self.pars_avd_home['choice'])
        self.pars_avd_home['choice'] = np.where(self.pars_avd_home['_11']==1,3,self.pars_avd_home['choice'])
        self.pars_avd_home.set_index(['region_id','iso_smaf','gr_age','choice'], inplace=True)
        self.pars_avd_home.drop(labels=['_10','_01','_11','_00'],inplace=True,axis=1)

        self.pars_inf_avq_rpa =  pd.read_csv(os.path.join(data_dir,'prob_csss_inf_avq_rpa.csv'),
            delimiter=';',low_memory=False)
        self.pars_inf_avq_rpa.set_index(['region_id','iso_smaf','gr_age'],inplace=True)
        self.pars_inf_avq_rpa.sort_index(inplace=True)
        self.pars_inf_avq_rpa = self.pars_inf_avq_rpa[['_00','_01','_10','_11']]
        self.pars_inf_avq_rpa.columns = ['inf_avq_00','inf_avq_01','inf_avq_10','inf_avq_11']

        self.pars_avd_rpa =  pd.read_csv(os.path.join(data_dir,'prob_csss_avd_rpa.csv'),
            delimiter=';',low_memory=False)
        self.pars_avd_rpa.columns = ['region_id','iso_smaf','gr_age','_10','_01','_11','_00', 'prob_avd']
        self.pars_avd_rpa = self.pars_avd_rpa[['region_id','iso_smaf','gr_age','_00','_01','_10','_11','prob_avd']]
        self.pars_avd_rpa['choice'] = 0
        self.pars_avd_rpa['choice'] = np.where(self.pars_avd_rpa['_00']==1,0,self.pars_avd_rpa['choice'])
        self.pars_avd_rpa['choice'] = np.where(self.pars_avd_rpa['_01']==1,1,self.pars_avd_rpa['choice'])
        self.pars_avd_rpa['choice'] = np.where(self.pars_avd_rpa['_10']==1,2,self.pars_avd_rpa['choice'])
        self.pars_avd_rpa['choice'] = np.where(self.pars_avd_rpa['_11']==1,3,self.pars_avd_rpa['choice'])
        self.pars_avd_rpa.set_index(['region_id','iso_smaf','gr_age','choice'], inplace=True)
        self.pars_avd_rpa.drop(labels=['_10','_01','_11','_00'],inplace=True,axis=1)

        return
    def assign(self, users, milieu):
        users['clsc_inf_any'] = False
        users['clsc_avq_any'] = False
        merge_key = ['region_id','iso_smaf','gr_age']
        if milieu == 'home':
            pars = self.pars_inf_avq_home
            pars_avd = self.pars_avd_home
            select = users.index[users.any_svc]
            work = users.loc[select, :]
        if milieu == 'rpa':
            pars = self.pars_inf_avq_rpa
            pars_avd = self.pars_avd_rpa
            work = users.copy()
        work = work.merge(pars, left_on=merge_key, right_on=merge_key, how='left')
        work['choice'] = draw_multinomial(work[['inf_avq_00','inf_avq_01','inf_avq_10','inf_avq_11']].values)
        work['clsc_inf_any'] = work['choice'].isin([2,3])
        work['clsc_avq_any'] = work['choice'].isin([1,3])
        merge_key = ['region_id', 'iso_smaf', 'gr_age','choice']
        work = work.merge(pars_avd,left_on=merge_key, right_on=merge_key, how='left')
        work['clsc_avd_any'] = np.random.uniform(size=len(work))<=work['prob_avd']
        users[['clsc_inf_any','clsc_avq_any','clsc_avd_any']] = False
        if milieu =='home':
            users.loc[select, ['clsc_inf_any','clsc_avq_any','clsc_avd_any']] = work[['clsc_inf_any','clsc_avq_any','clsc_avd_any']]
        if milieu =='rpa':
            users[['clsc_inf_any','clsc_avq_any','clsc_avd_any']] = work[['clsc_inf_any','clsc_avq_any','clsc_avd_any']]
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
    

@njit((int64[:])(float64[:,:]), cache=True, parallel=True)
def draw_multinomial(prob):
        n, m = prob.shape
        set = np.arange(m)
        result = np.zeros(n,dtype='int64')
        u = np.random.uniform(a=0.0,b=1.0,size=n)
        for i in prange(n):
            cp = 0.0
            for j  in range(m):
                cp += prob[i,j]
                if u[i]<=cp:
                    result[i] = j
                    break
        return result

