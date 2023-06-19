import pandas as pd
import numpy as np
import os
from itertools import product
from .needs import needs
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'SimSAD/data')
pd.options.mode.chained_assignment = None

class nsa:
    def __init__(self, open_capacity = 0.25):
        self.open_capacity = open_capacity
        return
    def load_register(self,start_yr=2019):
        reg = pd.read_csv(os.path.join(data_dir,'registre_nsa.csv'),
            delimiter=';',low_memory=False)
        reg = reg[reg.annee==start_yr]
        reg = reg[reg.region_id!=99]
        reg.set_index(['region_id'],inplace=True)
        reg.drop(labels='annee',axis=1,inplace=True)
        reg['nb_usagers'] = 0
        reg[['iso_smaf'+str(s) for s in range(1,15)]] = 0.0
        reg['nb_places'] *= self.open_capacity
        self.registry = reg
        self.days_per_year = 365
        return
    def assign(self,applicants,region_id):
        self.registry.loc[region_id,['iso_smaf'+str(s) for s in range(1,15)]] = applicants
        self.registry.loc[region_id,'nb_usagers'] = np.sum(applicants)
        return 
    def create_users(self, users):
        self.users = users.to_frame()
        self.users.columns = ['wgt']
        self.users.loc[self.users.wgt.isna(), 'wgt'] = 0.0
        self.users.wgt.clip(lower=0.0, inplace=True)
        self.users.wgt = self.users.wgt.astype('int64')
        self.users = self.users.reindex(self.users.index.repeat(self.users.wgt))
        self.users.wgt = 1
        self.users['smaf'] = self.users.index.get_level_values(1)
        self.users['milieu'] = 'ri'
        self.users['supplier'] = 'public'
        n = needs()
        for c in ['inf','avq','avd']:
            self.users['needs_'+c] = 0.0
        for s in range(1,15):
            self.users.loc[self.users.smaf==s,'needs_inf'] = n.inf[
                                                                 s-1]*self.days_per_year
            self.users.loc[self.users.smaf==s,'needs_avq'] = n.avq[
                                                                 s-1]*self.days_per_year
            self.users.loc[self.users.smaf==s,'needs_avd'] = n.avd[
                                                                 s-1]*self.days_per_year
        self.users['tx_serv_inf'] = 0.0
        self.users['tx_serv_avq'] = 0.0
        self.users['tx_serv_avd'] = 0.0
        self.users['wait_time'] = 0.0
        self.users['cost'] = 0.0
        self.users = self.users.reset_index()
        self.users['id'] = np.arange(len(self.users))
        self.users.set_index('id',inplace=True)
        return
    def reset_users(self):
        self.users = []
        return
    def compute_costs(self):
        self.registry['cout_total'] = self.registry['cout_place'] * self.registry['nb_usagers']
        return
    def update_users(self):
        self.users['tx_serv_inf'] = 100.0
        self.users['tx_serv_avq'] = 100.0
        self.users['tx_serv_avd'] = 100.0
        for r in range(1,19):
            self.users.loc[self.users.region_id==r,'cost'] = \
                self.registry.loc[r,'cah']
        self.users['cost'] *= 1/12
        return
    def collapse(self, domain = 'registry', rowvars=['region_id'],colvars=['smaf']):
        t = getattr(self, domain)
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