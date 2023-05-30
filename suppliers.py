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
        self.registry = self.registry[['region_id', 'heures_tot_trav_avd', 'nb_etc_avd']]
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
        users['pefsad_avd_hrs'] *= factor
        return users
    def compute_supply(self):
        self.registry['supply_avd'] = self.registry['nb_etc_avd'] * self.registry['hrs_per_etc']
        return
    def compute_serv_rate(self):
        self.registry['tx_svc_avd'] = self.registry['supply_avd'] / self.registry['needs_avd']
        self.registry['tx_svc_avd'].clip(upper=1.0, inplace=True)
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

class clsc:
    def __init__(self):
        self.care_types = ['inf','avq','avd']
        self.load_params()
        self.load_registry()
        return 
    def load_registry(self, start_yr = 2019):
        self.registry = pd.read_csv(os.path.join(data_dir, 'registre_clsc.csv'),
                                        delimiter=';', low_memory=False)
        self.registry = self.registry[self.registry.annee==start_yr]
        self.registry.set_index('region_id',inplace=True)
        self.days_per_year = 365

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
            regions = np.arange(1,19)
            smafs = np.arange(1,15)
            milieux = ['home','rpa','ri']
            tups = list(product(*[milieux, regions, smafs,self.care_types]))
            itups = pd.MultiIndex.from_tuples(tups)
            self.count = pd.DataFrame(index=itups, columns = ['users','hrs'],
                                      dtype = 'float64')
            self.count.loc[:,:] = 0.0
            self.count.sort_index(inplace=True)
            self.count.index.names = ['milieux','region_id','iso_smaf','svc']
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
        margins = ['clsc_inf_any', 'clsc_avq_any', 'clsc_avd_any','choice']
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
        users.sort_index(inplace=True)


        counts = pd.concat([users.groupby(['region_id','iso_smaf']).apply(
            lambda d: (d['clsc_'+c+'_any']*d['wgt']).sum()) for c in self.care_types],
                               axis=1)
        counts.columns = self.care_types
        counts = counts.stack()
        counts.index.names = ['region_id','iso_smaf','svc']
        counts = counts.reset_index()
        counts['milieu'] = milieu
        counts.set_index(['milieu','region_id','iso_smaf','svc'],inplace=True)
        counts.columns = ['users']
        hrs = pd.concat([users.groupby(['region_id',
                                               'iso_smaf']).apply(lambda d: (d[
                'clsc_'+c+'_hrs']*d['wgt']).sum()) for c in self.care_types],
                               axis=1)
        hrs.columns = self.care_types
        hrs = hrs.stack()
        hrs.index.names = ['region_id', 'iso_smaf', 'svc']
        hrs = hrs.reset_index()
        hrs['milieu'] = milieu
        hrs.set_index(['milieu','region_id','iso_smaf','svc'],inplace=True)
        hrs.index.names = ['milieu','region_id','iso_smaf','svc']
        hrs.columns = ['hrs']
        self.count.update(counts,join='left',overwrite=True)
        self.count.update(hrs, join='left', overwrite=True)

        #data = users.loc[select, ['clsc_' + c + '_any',
        #            'clsc_'+c+'_hrs','wgt']]

        #for c in self.care_types:
        #    for r in np.arange(1,19):
        #        for s in np.arange(1,15):
        #            tup = (milieu, r,  c, s)
        #            select = (users['region_id']==r) & (users['smaf']==s)
        #            data = users.loc[select,['clsc_'+c+'_any',
        #            'clsc_'+c+'_hrs','wgt']]
        #            self.count.loc[tup,'users'] = data.loc[data[
        #            'clsc_'+c+'_any'],'wgt'].sum(axis=0)
        #            self.count.loc[tup,'hrs'] = data[['clsc_'+c+'_hrs',
        #            'wgt']].prod(axis=1).sum(axis=0)
        return users
    def compute_needs(self):
        # use count to get distribution of needs for each care type
        users = pd.pivot_table(data=self.count,index=['region_id'],columns=[
            'svc','iso_smaf'],values='users',aggfunc='sum')
        n = needs()
        for c in self.care_types:
            self.registry['needs_' + c] = 0.0
        for s in range(1,15):
            self.registry['needs_inf'] += n.inf[s-1] * users.loc[:,('inf',s)]
            self.registry['needs_avq'] += n.avq[s-1] * users.loc[:,('avq',s)]
            self.registry['needs_avd'] += n.avd[s-1] * users.loc[:,('avd',s)]
        for c in self.care_types:
            self.registry['needs_' + c] *= self.days_per_year
        return


    def compute_supply(self):
        for c in self.care_types:
            self.registry['supply_'+c] = self.registry['nb_etc_'+c] * self.registry['hrs_moy_etc_'+c]
            if c!='avd':
                self.registry['supply_'+c] *= (1.0-self.registry['tx_hrs_indirect_'+c])
        return

    def cap(self, users):
        hrs_free = pd.pivot_table(data=self.count, index=['region_id'], columns=['svc'], values='hrs',
                               aggfunc='sum')
        for c in self.care_types:
            factor = self.registry['supply_'+c]/hrs_free[c]
            users['clsc_'+c+'_hrs'] *= factor
        return users

    def compute_serv_rate(self):
        for c in self.care_types:
            self.registry['tx_svc_'+c] = self.registry['supply_'+c]/self.registry['needs_'+c]
            self.registry['tx_svc_'+c].clip(upper=1.0,inplace=True)
        return
    def collapse_users(self,rowvars=['region_id','svc'],colvars=['iso_smaf']):
        table = pd.pivot_table(self.count.stack().to_frame(),index=rowvars,columns=colvars,aggfunc='sum')
        if colvars!=[]:
            table.columns = [x[1] for x in table.columns]
        return table
    def collapse(self, domain = 'registry', rowvars=['region_id'],colvars=['needs_inf']):
        t = getattr(self, domain)
        if domain == 'registry':
            if 'iso_smaf' in colvars:
                table = self.registry.loc[:,['iso_smaf'+str(s) for s in range(1,15)]]
                table.columns = [s for s in range(1,15)]
            else :
                table = self.registry.loc[:,colvars]
        if domain == 'users':
                table = pd.concat([self.users.groupby(rowvars).apply(lambda d: (d[c] * d.wgt).sum()) for c in colvars], axis=1)
                table.columns = colvars
                table = table[colvars]
        return table

# autres services achetés, AVQ + soins infirmiers     
class prive:
    def __init__(self):
        self.load_registry()
        return
    def load_registry(self, start_yr = 2019):
        self.registry = pd.read_csv(os.path.join(data_dir, 'registre_prive.csv'),
                                        delimiter=';', low_memory=False)
        self.registry = self.registry[self.registry.annee == start_yr]
        self.registry = self.registry[['region_id', 'hrs_avq', 'nb_etc_avq']]
        self.registry.set_index('region_id', inplace=True)
        self.registry['hrs_per_etc'] = self.registry['hrs_avq']/self.registry['nb_etc_avq']
        tups = list(product(*[np.arange(1,19),np.arange(1,15)]))
        itups = pd.MultiIndex.from_tuples(tups)
        itups.names = ['region_id','smaf']
        self.count = pd.DataFrame(index=itups,columns =['users','hrs'],dtype='float64')
        self.count.loc[:,:] = 0.0
        self.days_per_year = 365
        return
    def assign(self, users):
        for r in range(1,19):
            for s in range(1,15):
                select = (users['region_id']==r) & (users['smaf']==s) & (users['ces_any'])
                self.count.loc[(r,s),'users'] = users.loc[select,'wgt'].sum()
                self.count.loc[(r,s),'hrs'] = users.loc[select,['wgt','ces_hrs_avq']].prod(axis=1).sum()
        return

    def compute_needs(self):
        users = pd.pivot_table(data=self.count,index='region_id', columns='smaf', values='users',aggfunc='sum')
        n = needs()
        self.registry['needs_avq'] = 0.0
        for s in range(1,15):
            self.registry['needs_avq'] += n.avq[s-1] * users.loc[:,s]
        self.registry['needs_avq'] *= self.days_per_year
        return
    def cap(self, users):
        hrs_free = self.count.groupby('region_id').sum()['hrs']
        factor = self.registry['supply_avq']/hrs_free
        users['ces_hrs_avq'] *= factor
        return users
    def compute_supply(self):
        self.registry['supply_avq'] = self.registry['nb_etc_avq'] * self.registry['hrs_per_etc']
        return
    def compute_serv_rate(self):
        self.registry['tx_svc_avq'] = self.registry['supply_avq'] / self.registry['needs_avq']
        self.registry['tx_svc_avq'].clip(upper=1.0, inplace=True)
        return
    def collapse(self, domain = 'registry', rowvars=['region_id'],colvars=['needs_inf']):
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

