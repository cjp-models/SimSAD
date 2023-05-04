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
    def assign(self,applicants_none, applicants_svc, waiting_months, region_id):
        self.registry.loc[region_id,['iso_smaf_svc'+str(s) for s in range(1,15)]] = applicants_svc
        self.registry.loc[region_id,'nb_usagers_svc'] = np.sum(applicants_svc)
        self.registry.loc[region_id,['iso_smaf_none'+str(s) for s in range(1,15)]] = applicants_none
        self.registry.loc[region_id,'nb_usagers_none'] = np.sum(applicants_none)
        self.registry.loc[region_id,'attente_usagers_mois'] = waiting_months
        return
    def create_users(self, users_none, users_svc):
        # users with services
        users_svc = users_svc.to_frame()
        users_svc.columns = ['wgt']
        users_svc['any_svc'] = True
        users_svc.loc[users_svc.wgt.isna(), 'wgt'] = 0.0
        users_svc.wgt.clip(lower=0.0, inplace=True)
        users_svc = users_svc.reset_index()
        users_svc.set_index(['region_id','iso_smaf','gr_age','any_svc'], inplace = True)
        # users without services
        users_none = users_none.to_frame()
        users_none.columns = ['wgt']
        users_none['any_svc'] = False
        users_none.loc[users_none.wgt.isna(), 'wgt'] = 0.0
        users_none.wgt.clip(lower=0.0, inplace=True)
        users_none = users_none.reset_index()
        users_none.set_index(['region_id','iso_smaf','gr_age','any_svc'], inplace = True)
        self.users = pd.concat([users_svc,users_none],axis=0)
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
        self.users = self.users.reset_index()
        self.users['id'] = np.arange(len(self.users))
        self.users.set_index('id',inplace=True)
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
