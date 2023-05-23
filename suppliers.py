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

        # home
        self.pars_inf_avq =  pd.read_csv(os.path.join(data_dir,'prob_csss_inf_avq_home.csv'),
            delimiter=';',low_memory=False)
        self.pars_inf_avq['milieu'] = 'home'
        self.pars_inf_avq.set_index(['milieu','region_id','iso_smaf','gr_age','choice'],inplace=True)
        self.pars_inf_avq.sort_index(inplace=True)
        self.pars_inf_avq = self.pars_inf_avq.unstack()
        self.pars_inf_avq.columns = ['inf_avq_00','inf_avq_01','inf_avq_10','inf_avq_11']
        self.pars_avd =  pd.read_csv(os.path.join(data_dir,'prob_csss_avd_home.csv'),
            delimiter=';',low_memory=False)
        self.pars_avd['milieu'] = 'home'
        self.pars_avd.set_index(['milieu','region_id','iso_smaf','gr_age',
                                      'choice'],inplace=True)
        self.pars_avd.columns = ['prob_avd']
        self.pars_avd.sort_index(inplace=True)
        self.pars_hrs = pd.read_csv(os.path.join(data_dir,'hrs_csss_inf_avq_home.csv'),
            delimiter=';',low_memory=False)
        self.pars_hrs['milieu'] = 'home'
        self.pars_hrs.set_index(['milieu','region_id','iso_smaf','gr_age'],
                                         inplace=True)
        self.pars_hrs.sort_index(inplace=True)
        hrs_avd = pd.read_csv(os.path.join(data_dir,'hrs_csss_avd_home.csv'),
            delimiter=';',low_memory=False)
        hrs_avd['milieu'] = 'home'
        hrs_avd.set_index(['milieu','region_id','iso_smaf','gr_age'],
                                         inplace=True)
        hrs_avd.sort_index(inplace=True)
        self.pars_hrs = self.pars_hrs.merge(hrs_avd,left_index=True,
                                            right_index=True,how='left')

        for m in ['rpa','ri']:
            pars_inf_avq = pd.read_csv(
                os.path.join(data_dir, 'prob_csss_inf_avq_'+m+'.csv'),
                delimiter=';', low_memory=False)
            pars_inf_avq['milieu'] = m
            pars_inf_avq.set_index(
                ['milieu', 'region_id', 'iso_smaf', 'gr_age', 'choice'],
                inplace=True)
            pars_inf_avq.sort_index(inplace=True)
            pars_inf_avq = pars_inf_avq.unstack()
            pars_inf_avq.columns = ['inf_avq_00', 'inf_avq_01',
                                         'inf_avq_10', 'inf_avq_11']
            self.pars_inf_avq = pd.concat([self.pars_inf_avq,pars_inf_avq],
                                          axis=0)
            if m=='rpa':
                pars_avd = pd.read_csv(
                    os.path.join(data_dir, 'prob_csss_avd_home.csv'),
                    delimiter=';', low_memory=False)
                pars_avd['milieu'] = m
                pars_avd.set_index(
                    ['milieu', 'region_id', 'iso_smaf', 'gr_age',
                    'choice'], inplace=True)
                pars_avd.columns = ['prob_avd']
                pars_avd.sort_index(inplace=True)
                self.pars_avd = pd.concat([self.pars_avd,pars_avd],axis=0)

            pars_hrs = pd.read_csv(
                    os.path.join(data_dir, 'hrs_csss_inf_avq_'+m+'.csv'),
                    delimiter=';', low_memory=False)
            pars_hrs['milieu'] = m
            pars_hrs.set_index(['milieu', 'region_id', 'iso_smaf', 'gr_age'],
                    inplace=True)
            pars_hrs.sort_index(inplace=True)

            if m=='rpa':
                hrs_avd = pd.read_csv(os.path.join(data_dir,
                                'hrs_csss_avd_'+m+'.csv'), delimiter=';',
                                      low_memory=False)
                hrs_avd['milieu'] = m
                hrs_avd.set_index(['milieu', 'region_id', 'iso_smaf', 'gr_age'],
                              inplace=True)
                hrs_avd.sort_index(inplace=True)
                pars_hrs = pars_hrs.merge(hrs_avd, left_index=True,
                                                right_index=True, how='left')

            self.pars_hrs = pd.concat([self.pars_hrs,pars_hrs],axis=0)

        return
    def assign(self, users, milieu):
        # find parameters
        pars_inf_avq = self.pars_inf_avq.loc[
                       self.pars_inf_avq.index.get_level_values(
            0)==milieu,:]
        pars_inf_avq = pars_inf_avq.droplevel('milieu', axis=0)
        if milieu != 'ri':
            pars_avd = self.pars_avd.loc[self.pars_avd.index.get_level_values(
            0)==milieu,:]
            pars_avd = pars_avd.droplevel('milieu', axis=0)
        pars_hrs = self.pars_hrs.loc[self.pars_hrs.index.get_level_values(
            0)==milieu,:]
        pars_hrs = pars_hrs.droplevel('milieu', axis=0)
        merge_key = ['region_id', 'iso_smaf', 'gr_age']
        # select sample and initialize
        for c in self.care_types:
            users['clsc_'+c+'_any'] = False
            users['clsc_'+c+'_hrs'] = 0.0
        if milieu=='home':
            select = users.index[users.any_svc]
        else :
            select = users.index
        work = users.loc[select,:]

        # merge probability for INF and AVQ
        work = work.merge(pars_inf_avq, left_on=merge_key, right_on=merge_key,
                          how='left')
        # merge hours parameters
        work = work.merge(pars_hrs, left_on=merge_key, right_on=merge_key,
                          how='left')
        # find choice for INF and AVQ
        work['choice'] = draw_multinomial(work[['inf_avq_00', 'inf_avq_01',
                                                'inf_avq_10',
                                                'inf_avq_11']].values)
        # merge probability for AVD
        if milieu != 'ri':
            merge_key = ['region_id', 'iso_smaf', 'gr_age', 'choice']
            work = work.merge(pars_avd, left_on=merge_key, right_on=merge_key,
                              how='left')
        else :
            work['prob_avd'] = 0.0

        work['clsc_inf_any'] = work['choice'].isin([2, 3])
        work['clsc_avq_any'] = work['choice'].isin([1, 3])
        work['clsc_avd_any'] = np.random.uniform(size=len(work)) <= work[
            'prob_avd']
        margins = ['clsc_inf_any', 'clsc_avq_any', 'clsc_avd_any']
        users.loc[select, margins] = work[margins].copy()
        # hours
        for c in self.care_types:
            if c=='avd' and milieu=='ri':
                work['clsc_' + c + '_hrs'] = 0.0
            else :
                work['clsc_'+c+'_hrs'] = np.where(work['clsc_'+c+'_any'],
                                                      work['hrs_'+c],0.0)
        margins = ['clsc_'+c+'_hrs' for c in self.care_types]
        users.loc[select,margins] = work[margins].copy()
        return users

    def summary(self,home, rpa, ri):

        table = pd.DataFrame(index=['inf','avq','avd'],columns = ['home',
                                                                   'rpa','ri'])
        table.loc['inf','home'] = home.loc[home.clsc_inf_any,
        ['wgt','clsc_inf_hrs']].prod(axis=1).sum(axis=0)
        table.loc['inf','rpa'] = rpa.loc[rpa.clsc_inf_any,
        ['wgt','clsc_inf_hrs']].prod(axis=1).sum(axis=0)
        table.loc['inf','ri'] = ri.loc[ri.clsc_inf_any,
        ['wgt','clsc_inf_hrs']].prod(axis=1).sum(axis=0)

        table.loc['avq','home'] = home.loc[home.clsc_avq_any,
        ['wgt','clsc_avq_hrs']].prod(axis=1).sum(axis=0)
        table.loc['avq','rpa'] = rpa.loc[rpa.clsc_avq_any,
        ['wgt','clsc_avq_hrs']].prod(axis=1).sum(axis=0)
        table.loc['avq','ri'] = ri.loc[ri.clsc_avq_any,
        ['wgt','clsc_avq_hrs']].prod(axis=1).sum(axis=0)

        table.loc['avd','home'] = home.loc[home.clsc_avd_any,
        ['wgt','clsc_avq_hrs']].prod(axis=1).sum(axis=0)
        table.loc['avd','rpa'] = rpa.loc[rpa.clsc_avd_any,
        ['wgt','clsc_avd_hrs']].prod(axis=1).sum(axis=0)
        table.loc['avd', 'ri'] = 0.0

        return table

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

