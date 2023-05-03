import pandas as pd
import numpy as np
import os
from itertools import product
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'SimSAD/data')
pd.options.mode.chained_assignment = None


class home:
    def __init__(self):
        return
    def load_register(self,start_yr=2019):
        reg = pd.read_csv(os.path.join(data_dir,'registre_home.csv'),
            delimiter=';',low_memory=False)
        reg = reg[reg.annee==start_yr]
        reg = reg[reg.region_id!=99]
        reg.set_index(['region_id'],inplace=True)
        reg.drop(labels='annee',axis=1,inplace=True)
        # reset smaf allocation of patients
        self.registry = reg
        return
    def assign(self,applicants,waiting_months, region_id):
        self.registry.loc[region_id,['iso_smaf'+str(s) for s in range(1,15)]] = applicants
        self.registry.loc[region_id,'nb_usagers'] = np.sum(applicants)
        self.registry.loc[region_id,'attente_usagers_mois'] = waiting_months
        return
    def create_users(self, users):
        self.users = users.to_frame()
        self.users.columns = ['wgt']
        self.users.loc[self.users.wgt.isna(), 'wgt'] = 0.0
        self.users.wgt.clip(lower=0.0, inplace=True)
        self.users.wgt *= 0.1
        self.users.wgt = self.users.wgt.astype('int64')
        self.users = self.users.reindex(self.users.index.repeat(self.users.wgt))
        self.users.wgt = 10
        self.users['smaf'] = self.users.index.get_level_values(1)
        self.users['milieu'] = 'home'
        self.users['supplier'] = 'public'
        self.users['tx_serv_inf'] = 0.0
        self.users['tx_serv_avq'] = 0.0
        self.users['tx_serv_avd'] = 0.0
        self.users['wait_time'] = 0.0
        self.users['cost'] = 0.0
        return
    def reset_users(self):
        self.users = []
        return
    def collapse(self,rowvars=['region_id'],colvars=['smaf']):
        if 'smaf' in colvars:
            table = self.registry.loc[:,['iso_smaf'+str(s) for s in range(1,15)]]
            table.columns = [s for s in range(1,15)]
        else :
            table = self.registry.loc[:,colvars]
        return table
