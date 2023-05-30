import pandas as pd
import numpy as np
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)) , 'SimSAD/data')
pd.options.mode.chained_assignment = None


class chsld:
    def __init__(self, opt_build = False, opt_purchase = False):
        self.opt_build = opt_build
        self.opt_purchase = opt_purchase
        return
    def load_register(self,start_yr=2019):
        reg = pd.read_csv(os.path.join(data_dir,'registre_chsld.csv'),
            delimiter=';',low_memory=False)
        reg = reg[reg.annee==start_yr]
        reg = reg[reg.region_id!=99]
        reg.set_index(['region_id'],inplace=True)
        reg.drop(labels='annee',axis=1,inplace=True)
        keep_vars = ['nb_places_pub','nb_places_nc','nb_installations','nb_etc_avq','nb_etc_inf','heures_tot_trav_avq',
                   'heures_tot_trav_inf','nb_usagers_pub','nb_usagers_nc']
        for s in range(1,15):
            keep_vars.append('iso_smaf_pub'+str(s))
        for s in range(1,15):
            keep_vars.append('iso_smaf_nc'+str(s))
        reg = reg[keep_vars]
        # reset smaf allocation of patients
        self.registry = reg
        self.registry['tx_serv_inf'] = 0.0
        self.registry['tx_serv_avq'] = 0.0
        self.registry['tx_serv_avd'] = 100.0
        self.registry['attente_usagers'] = 0.0

        # nb places
        self.registry['nb_capacity_nc'] = self.registry.loc[:,'nb_places_nc']
        self.registry['nb_places_nc'] = self.registry.loc[:,'nb_usagers_nc']
        if self.opt_purchase==False:
            self.registry['nb_places_nc'] = 0
        self.registry['nb_places_tot'] = self.registry['nb_places_pub'] + self.registry['nb_places_nc']
        self.registry['nb_usagers_tot'] = self.registry['nb_usagers_pub'] + self.registry['nb_usagers_nc']
        for s in range(1,15):
            self.registry['iso_smaf_tot'+str(s)] = self.registry['iso_smaf_pub'+str(s)]*self.registry['nb_usagers_pub'] \
                                                   + self.registry['iso_smaf_nc'+str(s)]*self.registry['nb_places_nc']
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
        self.nmonths = 12
        self.weeks_per_year = (self.days_per_year/self.days_per_week)
        self.registry['hours_per_etc_inf'] = self.registry['heures_tot_trav_inf']/self.registry['nb_etc_inf']
        self.registry['hours_per_etc_avq'] = self.registry['heures_tot_trav_avq']/self.registry['nb_etc_avq']
        self.registry['etc_inf_per_installation'] = self.registry['nb_etc_inf']/self.registry['nb_installations']
        self.registry['etc_avq_per_installation'] = self.registry['nb_etc_avq']/self.registry['nb_installations']
        self.registry['places_per_installation'] = self.registry['nb_places_pub']/self.registry['nb_installations']

        return
    def assign(self, applicants, waiting_users, region_id):
        tot = (self.registry.loc[region_id, 'nb_places_pub'] +
               self.registry.loc[region_id, 'nb_places_nc'])
        if tot>0:
            share = self.registry.loc[region_id, 'nb_places_nc'] / (self.registry.loc[region_id, 'nb_places_pub'] +
                                                                self.registry.loc[region_id, 'nb_places_nc'])
        else :
            share = 0
        self.registry.loc[region_id, ['iso_smaf_tot' + str(s) for s in range(1, 15)]] = applicants
        self.registry.loc[region_id, ['iso_smaf_pub' + str(s) for s in range(1, 15)]] = applicants*(1-share)
        self.registry.loc[region_id, ['iso_smaf_nc' + str(s) for s in range(1, 15)]] = applicants*share
        self.registry.loc[region_id, 'nb_usagers_tot'] = np.sum(applicants)
        self.registry.loc[region_id,'nb_usagers_pub'] = np.sum(applicants) * (1.0-share)
        self.registry.loc[region_id,'nb_usagers_nc'] = np.sum(applicants) * share
        self.registry.loc[region_id,'attente_usagers'] = waiting_users
        return 
    def purchase(self, purchase_rate = 0.25):
        if self.opt_purchase:
            for r in self.registry.index:
                if self.registry.loc[r,'nb_places_nc']<self.registry.loc[r,'nb_capacity_nc']:
                    spots = self.registry.loc[r, 'nb_capacity_nc'] - self.registry.loc[r,'nb_places_nc']
                    self.registry.loc[r,'nb_places_nc'] += purchase_rate*min(self.registry.loc[r,'attente_usagers'],spots)
        self.registry['nb_places_tot'] = self.registry['nb_places_pub'] + self.registry['nb_places_nc']
        return
    def build(self, build_rate = 0.2):
        if self.opt_build:
            self.registry['nb_places_pub'] += (self.registry['attente_usagers'] * build_rate)
        self.registry['nb_places_tot'] = self.registry['nb_places_pub'] + self.registry['nb_places_nc']
        return 
    def compute_occupancy_rate(self):
        # occupancy rate
        self.registry['tx_occupation_pub'] = 100.0*(self.registry['nb_usagers_pub']/self.registry['nb_places_pub'])
        self.registry['tx_occupation_pub'] = self.registry['tx_occupation_pub'].clip(upper=100.0)
        return
    def compute_needs(self):
       # effective time per patient inf
        time_table = pd.DataFrame(index=self.registry.index,columns = np.arange(1,15),dtype='float64')
        for s in range(1,15):
            time_table[s] = self.registry['iso_smaf_pub'+str(s)]*self.needs_inf[s-1]
        time_per_usager_inf = time_table.sum(axis=1)
        time_per_usager_inf += self.inf_indirect_per_day*self.registry['nb_usagers_pub']
        time_per_usager_inf *= self.days_per_year
        # effective time per patient avq
        for s in range(1,15):
            time_table[s] = self.registry['iso_smaf_pub'+str(s)]*self.needs_avq[s-1]
        time_per_usager_avq = time_table.sum(axis=1)
        time_per_usager_avq *= self.days_per_year
        result = pd.concat([time_per_usager_inf,time_per_usager_avq],axis=1)
        result.columns = ['inf','avq']
        return result
    def compute_supply(self):
        # inf
        time_inf = self.registry['hours_per_etc_inf'].copy()
        # take out pauses
        time_inf -= self.time_per_pause*(self.weeks_per_year-self.weeks_vacation_inf)*self.work_days_per_week
        # take out indirect care
        time_inf *= (1.0 - self.share_indirect_care)
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
        self.users['milieu'] = 'chsld'
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
    def update_users(self):
        # merge tx service on region
        return
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
