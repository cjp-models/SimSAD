import pandas as pd
import numpy as np
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'SimSAD/data')


class ces:
    def __init__(self):
        self.load_params()
        return
    def load_params(self, start_yr = 2021):
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


