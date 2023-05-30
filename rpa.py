import pandas as pd
import numpy as np
import os
from itertools import product
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'SimSAD/data')
pd.options.mode.chained_assignment = None

class rpa:
    def __init__(self, opt_build = False):
        self.opt_build = False
        return
    def load_register(self,start_yr=2019):
        reg = pd.read_csv(os.path.join(data_dir,'registre_rpa.csv'),
            delimiter=';',low_memory=False)
        reg = reg[reg.annee==start_yr]
        reg = reg[reg.region_id!=99]
        reg.set_index(['region_id'],inplace=True)
        reg.drop(labels='annee',axis=1,inplace=True)
        keep_vars = ['nb_places','nb_installations','nb_usagers_sad','tx_occupation']
        for s in range(1,15):
            keep_vars.append('iso_smaf'+str(s))
        reg = reg[keep_vars]
        reg.rename({'nb_usagers_sad':'nb_usagers'},axis=1,inplace=True)
        reg['nb_places'] = reg.loc[:,'nb_usagers']
        # reset smaf allocation of patients
        self.registry = reg
        self.registry['tx_serv_inf'] = 0.0
        self.registry['tx_serv_avq'] = 0.0
        self.registry['tx_serv_avd'] = 0.0
        # needs weights (hours per day of care by smaf, Tousignant)
        self.needs_inf = [0.01,0.02,0.23,0.15,0.29,0.31,0.33,
                            0.38,0.43,0.48,0.58,0.47,0.69,0.95]
        self.needs_avq = [0.26,0.27,0.48,0.57,0.67,0.68,
                            1.08,1.24,2.29,2.29,2.61,2.54,2.62,3.08]
        self.inf_indirect_per_day = 0.4
        self.days_per_week = 7
        self.days_per_year = 365
        self.weeks_vacation_inf = 4
        self.weeks_vacation_avq = 4
        self.work_days_per_week = 5
        self.share_indirect_care = 0.2
        self.time_per_pause = 0.5
        self.nsmafs = 14
        self.nmonths = 24
        self.weeks_per_year = (self.days_per_year/self.days_per_week)
        self.registry['nb_etc_inf'] = 0.3 * self.registry['nb_places']
        self.registry['nb_etc_avq'] = 0.6 * self.registry['nb_places']
        self.registry['heures_tot_trav_inf'] = self.registry['nb_etc_inf'] * 1500
        self.registry['heures_tot_trav_avq'] = self.registry['nb_etc_avq'] * 1500
        self.registry['attente_usagers_mois'] = 0.0
        self.registry['hours_per_etc_inf'] = self.registry['heures_tot_trav_inf']/self.registry['nb_etc_inf']
        self.registry['hours_per_etc_avq'] = self.registry['heures_tot_trav_avq']/self.registry['nb_etc_avq']
        self.registry['etc_inf_per_installation'] = self.registry['nb_etc_inf']/self.registry['nb_installations']
        self.registry['etc_avq_per_installation'] = self.registry['nb_etc_avq']/self.registry['nb_installations']
        self.registry['places_per_installation'] = self.registry['nb_places']/self.registry['nb_installations']
        return
    def assign(self,applicants,waiting_months,region_id):
        self.registry.loc[region_id,['iso_smaf'+str(s) for s in range(1,15)]] = applicants
        self.registry.loc[region_id,'nb_usagers'] = np.sum(applicants)
        self.registry.loc[region_id,'attente_usagers_mois'] = waiting_months
        return 
    def build(self):
        if self.opt_build:
            work = self.registry[['nb_installations','places_per_installation','attente_usagers_mois','nb_places']].copy()
            work['attente_usagers'] = work['attente_usagers_mois']/12.0
            self.registry['nb_places'] += work['attente_usagers']
        return 
    def compute_occupancy_rate(self):
        # occupancy rate
        self.registry['tx_occupation'] = 100.0*(self.registry['nb_usagers']/self.registry['nb_places'])
        self.registry['tx_occupation'] = self.registry['tx_occupation'].clip(upper=100.0)
        return
    def compute_needs(self):
       # effective time per patient inf
        time_table = pd.DataFrame(index=self.registry.index,columns = np.arange(1,15),dtype='float64')
        for s in range(1,15):
            time_table[s] = self.registry['iso_smaf'+str(s)]*self.needs_inf[s-1]
        time_per_usager_inf = time_table.sum(axis=1)
        time_per_usager_inf += self.inf_indirect_per_day*self.registry['nb_usagers']
        time_per_usager_inf *= self.days_per_year
        # effective time per patient avq
        for s in range(1,15):
            time_table[s] = self.registry['iso_smaf'+str(s)]*self.needs_avq[s-1]
        time_per_usager_avq = time_table.sum(axis=1)
        time_per_usager_avq *= self.days_per_year
        result = pd.concat([time_per_usager_inf,time_per_usager_avq],axis=1)
        result.columns = ['inf','avq']
        return result
    def compute_supply(self):
        # inf
        time_inf = self.registry['hours_per_etc_inf'].copy()
        # take out indirect care
        time_inf *= (1.0 - self.share_indirect_care)
        # take out pauses
        time_inf -= self.time_per_pause*(self.weeks_per_year-self.weeks_vacation_inf)*self.work_days_per_week
        # blow up using number of nurses
        time_inf  = time_inf * self.registry['nb_etc_inf']
        ## avq
        time_avq = self.registry['hours_per_etc_avq'].copy()
        # take out pauses
        time_avq -= self.time_per_pause*(self.weeks_per_year-self.weeks_vacation_avq)*self.work_days_per_week
        # blow up using number of nurses
        time_avq  = time_avq * self.registry['nb_etc_avq']
        result = pd.concat([time_inf,time_avq],axis=1)
        result.columns = ['inf','avq']
        return result
    def compute_serv_rate(self):
        self.compute_occupancy_rate()
        needs = self.compute_needs()
        supply = self.compute_supply()
        self.registry['heures_tot_trav_inf'] = supply['inf']
        self.registry['heures_tot_trav_avq'] = supply['avq']
        self.registry['tx_serv_inf'] = np.where(needs['inf']>0,100.0*(supply['inf']/needs['inf']),np.nan)
        self.registry.loc[self.registry.tx_serv_inf>100.0,'tx_serv_inf'] = 100.0
        self.registry['tx_serv_avq'] = np.where(needs['avq']>0,100.0*(supply['avq']/needs['avq']),np.nan)
        self.registry.loc[self.registry.tx_serv_avq>100.0,'tx_serv_avq']= 100.0
        return
    def create_users(self, users):
        self.users = users.to_frame()
        self.users.columns = ['wgt']
        self.users.loc[self.users.wgt.isna(), 'wgt'] = 0.0
        self.users.wgt.clip(lower=0.0, inplace=True)
        self.users.wgt = self.users.wgt.astype('int64')
        self.users.wgt *= 0.25
        self.users = self.users.reindex(self.users.index.repeat(self.users.wgt))
        self.users.wgt = 4
        self.users['smaf'] = self.users.index.get_level_values(1)
        self.users['milieu'] = 'rpa'
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