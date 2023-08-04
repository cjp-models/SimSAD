import pandas as pd 
import numpy as np 
from itertools import product
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'SimSAD/data')
from numba import njit, float64, int64, boolean, prange
from numba.types import Tuple
from .needs import needs
from itertools import product




# autres services achetés, AVQ + soins infirmiers     
class prive:
    def __init__(self, policy):
        self.load_registry()
        self.policy = policy
        return
    def load_registry(self, start_yr = 2021):
        self.registry = pd.read_csv(os.path.join(data_dir, 'registre_prive.csv'),
                                        delimiter=';', low_memory=False)
        self.registry = self.registry[self.registry.annee == start_yr]
        self.registry.set_index('region_id', inplace=True)
        self.registry['hrs_per_etc_avq'] = self.registry[
                                            'heures_tot_trav_avq']/self.registry['nb_etc_avq']
        self.registry['hrs_per_etc_avd'] = self.registry[
                                            'heures_tot_trav_avd']/self.registry[
            'nb_etc_avd']
        tups = list(product(*[np.arange(1,19),np.arange(1,15)]))
        itups = pd.MultiIndex.from_tuples(tups)
        itups.names = ['region_id','smaf']
        self.count = pd.DataFrame(index=itups,columns =['users','hrs_avq',
                                                        'hrs_avd'],
                                  dtype='float64')
        self.count.loc[:,:] = 0.0
        self.days_per_year = 365
        return
    def assign(self, users):
        for r in range(1,19):
            for s in range(1,15):
                select = (users['region_id']==r) & (users['smaf']==s) & (users['ces_any'])
                self.count.loc[(r,s),'users'] = users.loc[select,'wgt'].sum()
                self.count.loc[(r,s),'hrs_avq'] = users.loc[select,['wgt',
                                                                 'ces_hrs_avq']].prod(axis=1).sum()
                self.count.loc[(r,s),'hrs_avd'] = users.loc[select,['wgt',
                                                                 'ces_hrs_avd']].prod(axis=1).sum()
        return
    def cap(self, users):
        hrs_free = self.count.groupby('region_id').sum()['hrs_avq']
        factor = self.registry['supply_avq']/hrs_free
        factor.clip(upper=1.0,inplace=True)
        factor[factor.isna()] = 1.0
        excess = hrs_free - self.registry['supply_avq']
        excess.clip(lower=0.0, inplace=True)
        excess[excess.isna()] = 0.0
        indirect = (1.0 - self.registry['tx_hrs_dep_avq'])
        excess = excess / indirect
        excess = excess / self.registry['hrs_per_etc_avq']
        self.registry['worker_needs_avq'] = excess
        self.registry.loc[self.registry.worker_needs_avq.isna(),
                'worker_needs_avq'] = 0.0
        for r in range(1,19):
            users.loc[users.region_id==r,'ces_hrs_avq'] *= factor[r]

        hrs_free = self.count.groupby('region_id').sum()['hrs_avd']
        factor = self.registry['supply_avd']/hrs_free
        factor.clip(upper=1.0,inplace=True)
        factor[factor.isna()] = 1.0
        excess = hrs_free - self.registry['supply_avd']
        excess.clip(lower=0.0, inplace=True)
        excess[excess.isna()] = 0.0
        indirect = (1.0 - self.registry['tx_hrs_dep_avd'])
        excess = excess / indirect
        excess = excess / self.registry['hrs_per_etc_avd']
        self.registry['worker_needs_avd'] = excess
        self.registry.loc[self.registry.worker_needs_avd.isna(),
                'worker_needs_avd'] = 0.0
        for r in range(1,19):
            users.loc[users.region_id==r,'ces_hrs_avq'] *= factor[r]
        return users
    def compute_supply(self):
        self.registry['supply_avq'] = self.registry['nb_etc_avq'] * \
                                      self.registry['hrs_per_etc_avq'] * (1.0
                                                                          -
                                                                          self.registry['tx_hrs_dep_avq'])
        self.registry['supply_avd'] = self.registry['nb_etc_avd'] * \
                                      self.registry['hrs_per_etc_avd']*(1.0
                                                                          -
                                                                          self.registry['tx_hrs_dep_avd'])

        return
    def compute_costs(self):
        self.registry['cout_fixe'] = 0.0
        self.registry['cout_var'] = self.registry['sal_avq'] * self.registry[
            'nb_etc_avq'] * self.registry['hrs_per_etc_avq']
        self.registry['cout_var'] += self.registry['sal_avd'] * self.registry[
            'nb_etc_avd'] * self.registry['hrs_per_etc_avd']
        self.registry['cout_total'] = self.registry['cout_var'] + self.registry['cout_fixe']
        return
    def workforce(self,before_base_yr=False):
        for c in ['avd','avq']:
            if before_base_yr:
                attr = 1.0
            else:
                attr = getattr(self.policy,'prive_'+c+'_rate')
            self.registry['nb_etc_'+c] += \
                        attr * self.registry['worker_needs_'+c]
        return
    def collapse(self, domain = 'registry', rowvars=['region_id'],colvars=['needs_inf']):
        if domain == 'registry':
            if 'smaf' in colvars:
                table = self.registry.loc[:,['iso_smaf'+str(s) for s in range(1,15)]]
                table.columns = [s for s in range(1,15)]
            else :
                table = self.registry.loc[:,colvars]
        if domain == 'users':
                table = pd.concat([self.users.groupby(rowvars).apply(lambda d: (d[c] * d.wgt).sum()) for c in colvars], axis=1)
                table.columns = colvars
                table = table[colvars]
        return table
    

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

