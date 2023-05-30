import pandas as pd 
import numpy as np 
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'SimSAD/data')

# PEFSAD
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
# CES
class ces:
    def __init__(self):
        self.load_params()
        return
    def load_params(self, start_yr = 2019):
        self.prob =  pd.read_csv(os.path.join(data_dir,'prob_ces.csv'),
            delimiter=';',low_memory=False)
        self.prob = self.prob[self.prob.annee==start_yr]
        self.prob = self.prob[self.prob.region_id!=99]
        self.prob = self.prob.drop(labels=['annee'],axis=1)
        self.prob.set_index(['region_id','iso_smaf','gr_age'],inplace=True)
        self.prob.sort_index(inplace=True)
        self.hrs = pd.read_csv(os.path.join(data_dir,'hrs_ces.csv'),
            delimiter=';',low_memory=False)
        self.hrs = self.hrs[self.hrs.annee==start_yr]
        self.hrs = self.hrs[self.hrs.region_id!=99]
        self.hrs = self.hrs.drop(labels=['annee'],axis=1)
        self.hrs.set_index(['region_id','iso_smaf','gr_age'],inplace=True)
        self.hrs.sort_index(inplace=True)
        return
    def assign(self,users):
        merge_key = ['region_id','iso_smaf','gr_age']
        work = users.copy()
        work = work.merge(self.prob,left_on = merge_key,right_on = merge_key,
                   how = 'left')
        work['u'] = np.random.uniform(size=len(work))
        work['ces_any'] = (work['u']<=work['prob'])

        work = work.merge(self.hrs,left_on = merge_key,right_on = merge_key,
                   how = 'left')
        work['ces_hrs_avq'] = 0
        work['ces_hrs_avd'] = 0
        work.loc[work.ces_any,'ces_hrs_avq'] = work.loc[work.ces_any, 'hrs_avq']
        work.loc[work.ces_any, 'ces_hrs_avd'] = work.loc[
            work.ces_any, 'hrs_avd']
        users[['ces_any','ces_hrs_avd','ces_hrs_avq']] = work.loc[:,['ces_any',
                                                                'ces_hrs_avd','ces_hrs_avq']]
        #print(users[['ces_any', 'ces_hrs_avd', 'ces_hrs_avq']].sum(axis=0))
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

# SAD credit (maison et RPA)   
class cmd:
    def __init__(self):
        self.load_register()
        return
    def load_register(self, start_yr = 2019):
        self.register =  pd.read_csv(os.path.join(data_dir,'mnt_cmd.csv'),
            delimiter=';',low_memory=False)
        self.register = self.register[self.register['annee']==start_yr]
        self.register['gr_age'] = 3
        register_frame = self.register.copy()
        for x in [1, 2]:
            register_frame['gr_age'] = x
            self.register = pd.concat([self.register,register_frame])
        self.register.set_index(['region_id', 'iso_smaf', 'gr_age'], inplace=True)
        self.register = self.register[['dom','rpa']]
        self.register.columns = ['mnt_home','mnt_rpa']
        self.register.loc[self.register.index.get_level_values(2)==1,:] = 0.0
        self.register.loc[self.register.index.get_level_values(2)==2,:] = 0.0
        return
    def assign(self, users, milieu):
        work = users.copy()
        merge_keys = ['region_id','iso_smaf','gr_age']
        work = work.merge(self.register,left_on=merge_keys,right_on=merge_keys,how='left')
        users['cmd_mnt'] = work['mnt_'+milieu]
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

# financement public SAD
class msss:
    def __init__(self):
        return 


