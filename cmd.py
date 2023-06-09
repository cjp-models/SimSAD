import pandas as pd 
import numpy as np 
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'SimSAD/data')


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


