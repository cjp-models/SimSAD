import pandas as pd
import numpy as np
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'SimSAD/data')


class pefsad:
    def __init__(self):
        self.load_params()
        return
    def load_params(self, start_yr = 2019):
        self.pars_home = pd.read_csv(os.path.join(data_dir,'pefsad_home.csv'),
            delimiter=';',low_memory=False)
        self.pars_home['clsc_avd_any'] = (self.pars_home['AVD']==1)
        self.pars_home.drop(labels=['AVD'],axis=1,inplace=True)
        self.pars_home.set_index(['region_id','iso_smaf','gr_age','choice','clsc_avd_any'],inplace=True)
        self.pars_home.columns = ['prob','hrs']
        self.pars_rpa = pd.read_csv(os.path.join(data_dir,'pefsad_rpa.csv'),
            delimiter=';',low_memory=False)
        self.pars_rpa['clsc_avd_any'] = (self.pars_rpa['AVD'] ==1)
        self.pars_rpa.drop(labels=['AVD'], axis=1, inplace=True)
        self.pars_rpa.set_index(['region_id','iso_smaf','gr_age','choice','clsc_avd_any'],inplace=True)
        self.pars_rpa.columns = ['prob', 'hrs']
        return
    def assign(self, users, milieu):
        merge_key = ['region_id','iso_smaf','gr_age','choice','clsc_avd_any']
        work = users.copy()
        if milieu=='home':
            work = work.merge(self.pars_home,left_on = merge_key,right_on = merge_key,
                   how = 'left')
        if milieu=='rpa':
            work = work.merge(self.pars_rpa,left_on = merge_key,right_on = merge_key,
                   how = 'left')
        work['u'] = np.random.uniform(size=len(work))
        work['pefsad_avd_any'] = (work['u']<work['prob'])
        work['pefsad_avd_hrs'] = 0
        work.loc[work.pefsad_avd_any,'pefsad_avd_hrs'] = work.loc[work.pefsad_avd_any, 'hrs']
        users[['pefsad_avd_any','pefsad_avd_hrs']] = work[['pefsad_avd_any','pefsad_avd_hrs']]
        return users
    def collapse(self, domain = 'registry', rowvars=['region_id'],colvars=['smaf']):
        t = getattr(self, domain)
        if domain == 'registry':
            if 'smaf' in colvars:
                table = self.registry.loc[:,['iso_smaf_tot'+str(s) for s in range(1,15)]]
                table.columns = [s for s in range(1,15)]
            else :
                table = self.registry.loc[:,colvars]
        if domain == 'users':
                table = pd.concat([self.users.groupby(rowvars).apply(lambda d: (d[c] * d.wgt).sum()) for c in colvars], axis=1)
                table.columns = colvars
                table = table[colvars]
        return table