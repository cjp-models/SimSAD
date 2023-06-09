import pandas as pd
import numpy as np
from itertools import product
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'SimSAD/data')
from numba import njit, float64, int64, boolean, prange
from numba.types import Tuple
from .needs import needs
from itertools import product



class eesad:
    def __init__(self):
        self.load_registry()
        return
    def load_registry(self, start_yr = 2019):
        self.registry = pd.read_csv(os.path.join(data_dir, 'registre_eesad.csv'),
                                        delimiter=';', low_memory=False)
        self.registry = self.registry[self.registry.annee == start_yr]
        self.registry.set_index('region_id', inplace=True)
        self.registry['hrs_per_etc'] = self.registry['heures_tot_trav_avd']/self.registry['nb_etc_avd']
        tups = list(product(*[np.arange(1,19),np.arange(1,15)]))
        itups = pd.MultiIndex.from_tuples(tups)
        itups.names = ['region_id','smaf']
        self.count = pd.DataFrame(index=itups,columns =['users','hrs'],dtype='float64')
        self.count.loc[:,:] = 0.0
        self.days_per_year = 365
        return
    def assign(self, users_home, users_rpa):
        for r in range(1,19):
            for s in range(1,15):
                select = (users_home['region_id']==r) & (users_home['smaf']==s) & (users_home['pefsad_avd_any'])
                self.count.loc[(r,s),'users'] = users_home.loc[select,'wgt'].sum()
                self.count.loc[(r,s),'hrs'] = users_home.loc[select,['wgt','pefsad_avd_hrs']].prod(axis=1).sum()
                select = (users_rpa['region_id']==r) & (users_rpa['smaf']==s) & (users_rpa['pefsad_avd_any'])
                self.count.loc[(r,s),'users'] += users_rpa.loc[select,'wgt'].sum()
                self.count.loc[(r,s),'hrs'] += users_rpa.loc[select,['wgt','pefsad_avd_hrs']].prod(axis=1).sum()
        return

    def compute_needs(self):
        users = pd.pivot_table(data=self.count,index='region_id', columns='smaf', values='users',aggfunc='sum')
        n = needs()
        self.registry['needs_avd'] = 0.0
        for s in range(1,15):
            self.registry['needs_avd'] += n.avd[s-1] * users.loc[:,s]
        self.registry['needs_avd'] *= self.days_per_year
        return
    def cap(self, users):
        hrs_free = self.count.groupby('region_id').sum()['hrs']
        factor = self.registry['supply_avd']/hrs_free
        for r in range(1,19):
            users.loc[users.region_id==r,'pefsad_avd_hrs'] *= min(factor[r],1.0)


        return users
    def compute_supply(self):
        self.registry['supply_avd'] = self.registry['nb_etc_avd'] * self.registry['hrs_per_etc']
        return
    def compute_serv_rate(self):
        self.registry['tx_svc_avd'] = self.registry['supply_avd'] / self.registry['needs_avd']
        self.registry['tx_svc_avd'].clip(upper=1.0, inplace=True)
        return
    def compute_costs(self):
        self.registry['cout_fixe'] = 0.0
        self.registry['cout_var'] = self.registry['sal_avd'] * self.registry['nb_etc_avd'] * self.registry['hrs_per_etc']
        self.registry['cout_total'] = self.registry['cout_fixe'] + self.registry['cout_var']
        return
    def collapse(self, domain = 'registry', rowvars=['region_id'],colvars=['needs_inf']):
        t = getattr(self, domain)
        if domain == 'registry':
            if 'smaf' in colvars:
                table = self.registry.loc[:,['iso_smaf_svc'+str(s) for s in range(1,15)]]
                table.columns = [s for s in range(1,15)]
            else :
                table = self.registry.loc[:,colvars]
        if domain == 'users':
                table = pd.concat([self.users.groupby(rowvars).apply(lambda d: (d[c] * d.wgt).sum()) for c in colvars], axis=1)
                table.columns = colvars
                table = table[colvars]
        return table

