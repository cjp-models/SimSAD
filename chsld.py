import pandas as pd
import numpy as np
import os
from .needs import needs
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)) , 'SimSAD/data')
pd.options.mode.chained_assignment = None


class chsld:
    def __init__(self, policy):
        self.policy = policy
        self.opt_build = self.policy.chsld_build
        self.build_rate = self.policy.chsld_build_rate
        self.opt_purchase = self.policy.chsld_purchase
        self.purchase_rate = self.policy.chsld_purchase_rate
        self.opt_mda = self.policy.chsld_mda
        self.infl_construction = self.policy.infl_construction
        self.interest_rate = self.policy.interest_rate
        self.chsld_inf_rate = self.policy.chsld_inf_rate
        self.chsld_avq_rate = self.policy.chsld_avq_rate
        return
    def load_register(self,start_yr=2019):
        reg = pd.read_csv(os.path.join(data_dir,'registre_chsld.csv'),
            delimiter=';',low_memory=False)
        reg = reg[reg.annee==start_yr]
        reg = reg[reg.region_id!=99]
        reg.set_index(['region_id'],inplace=True)
        reg.drop(labels='annee',axis=1,inplace=True)
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
        n = needs()
        self.needs_inf = n.inf
        self.needs_avq = n.avq

        self.inf_indirect_per_day = 0.4
        self.days_per_week = 7
        self.days_per_year = 365
        self.hours_per_day = 7.5
        self.share_indirect_care = 0.2
        self.time_per_pause = 0.5
        self.nsmafs = 14
        self.amort_rate = 1/25
        self.cost_rate = self.interest_rate + self.amort_rate
        self.weeks_per_year = (self.days_per_year/self.days_per_week)
        self.registry['hours_per_etc_inf'] = self.registry['heures_tot_trav_inf']/self.registry['nb_etc_inf']
        self.registry['hours_per_etc_avq'] = self.registry['heures_tot_trav_avq']/self.registry['nb_etc_avq']
        self.registry['etc_inf_per_usager'] = self.registry[
                                             'nb_etc_inf']/self.registry[
            'nb_usagers_pub']
        self.registry['etc_avq_per_usager'] = self.registry[
                                             'nb_etc_avq']/self.registry[
            'nb_usagers_pub']
        self.registry['places_per_installation'] = self.registry['nb_places_pub']/self.registry['nb_installations']
        self.registry['nb_build'] = 0
        if self.opt_mda:
            self.registry['cout_construction'] = self.registry['cout_construction_mda']
        else :
            self.registry['cout_construction'] = self.registry['cout_construction_chsld']
        self.registry['cout_comptable_immo'] = 0.0
        return
    def assign(self, applicants, waiting_users, region_id):
        tot = (self.registry.loc[region_id, 'nb_places_pub'] +
               self.registry.loc[region_id, 'nb_places_nc'])
        if tot>0:
            share = self.registry.loc[region_id, 'nb_places_nc'] / tot
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
    def purchase(self):
        if self.opt_purchase:
            for r in self.registry.index:
                if self.registry.loc[r,'nb_places_nc']<self.registry.loc[r,'nb_capacity_nc']:
                    spots = self.registry.loc[r, 'nb_capacity_nc'] - self.registry.loc[r,'nb_places_nc']
                    self.registry.loc[r,'nb_places_nc'] += \
                        self.purchase_rate*min(self.registry.loc[r,
                        'attente_usagers'],spots)
        self.registry['nb_places_tot'] = self.registry['nb_places_pub'] + self.registry['nb_places_nc']
        return
    def build(self):
        if self.opt_build:
            self.registry['cout_construction'] *= (1.0 + self.infl_construction)
            build = self.registry['attente_usagers'] * self.build_rate
            self.registry['attente_usagers'] -= build
            self.registry['cout_comptable_immo'] += build * self.registry['cout_construction']
            self.registry['nb_places_pub'] += build
            self.registry['nb_etc_inf'] += self.chsld_inf_rate * self.registry[
                'etc_inf_per_usager'] * build
            self.registry['nb_etc_avq'] += self.chsld_avq_rate * self.registry[
                'etc_avq_per_usager'] * build
        self.registry['nb_places_tot'] = self.registry['nb_places_pub'] + self.registry['nb_places_nc']
        return
    def compute_supply(self):
        # inf
        time_inf = self.registry['hours_per_etc_inf'].copy()
        # take out pauses self.hours_per_day
        time_inf -= self.time_per_pause*(time_inf/self.hours_per_day)
        # take out indirect care
        time_inf *= (1.0 - self.share_indirect_care)
        # blow up using number of nurses
        time_inf  = time_inf * self.registry['nb_etc_inf']
        ## avq
        time_avq = self.registry['hours_per_etc_avq'].copy()
        # take out pauses
        time_avq -= self.time_per_pause*(time_avq/self.hours_per_day)
        # blow up using number of avq workers
        time_avq  = time_avq * self.registry['nb_etc_avq']
        result = pd.concat([time_inf,time_avq],axis=1)
        result.columns = ['inf','avq']
        self.registry['supply_inf'] = result['inf']
        self.registry['supply_avq'] = result['avq']
        return result
    def compute_costs(self):
        self.registry['heures_tot_trav_inf'] = self.registry['nb_etc_inf'] * \
                                               self.registry['hours_per_etc_inf']
        self.registry['heures_tot_trav_avq'] = self.registry['nb_etc_avq'] * \
                                               self.registry['hours_per_etc_avq']
        self.registry['cout_inf'] = self.registry['sal_inf'] * self.registry['heures_tot_trav_inf']
        self.registry['cout_avq'] = self.registry['sal_avq'] * self.registry['heures_tot_trav_avq']
        self.registry['cout_var'] = self.registry['cout_inf'] + self.registry['cout_avq']
        self.registry['cout_var'] += self.registry['cout_place_var'] * \
                                     self.registry['nb_usagers_nc']
        self.registry['cout_fixe'] = self.registry['cout_place_fixe'] * self.registry['nb_usagers_tot']
        self.registry['cout_immo'] = self.registry['cout_comptable_immo']*self.cost_rate
        self.registry['cout_total'] = self.registry['cout_fixe'] + self.registry['cout_var'] + self.registry['cout_immo']
        self.registry['cout_place_var'] = self.registry['cout_var']/self.registry['nb_usagers_tot']
        self.registry['cout_place_total'] = self.registry['cout_total']/self.registry['nb_usagers_tot']
        return
    def create_users(self, users):
        self.users = users.to_frame()
        self.users.columns = ['wgt']
        self.users.loc[self.users.wgt.isna(), 'wgt'] = 0.0
        self.users.wgt.clip(lower=0.0, inplace=True)
        self.users.wgt = self.users.wgt.astype('int64')
        sample_ratio = 0.25
        self.users.wgt *= sample_ratio
        self.users = self.users.reindex(self.users.index.repeat(self.users.wgt))
        self.users.wgt = 1/sample_ratio
        self.users['smaf'] = self.users.index.get_level_values(1)
        self.users['milieu'] = 'chsld'
        self.users['supplier'] = 'public'
        n = needs()
        for c in ['inf','avq','avd']:
            self.users['needs_'+c] = 0.0
        for s in range(1,15):
            self.users.loc[self.users.smaf==s,'needs_inf'] = n.inf[
                                                                 s-1]*self.days_per_year
            self.users.loc[self.users.smaf==s,'needs_avq'] = n.avq[s-1]*self.days_per_year
            self.users.loc[self.users.smaf==s,'needs_avd'] = n.avd[s-1]*self.days_per_year
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
        # get how many hours are supplied for each domain
        hrs_per_users_inf = self.registry['supply_inf']/self.registry[
            'nb_usagers_pub']
        hrs_per_users_avq = self.registry['supply_avq']/self.registry[
            'nb_usagers_pub']
        self.users[['serv_inf','serv_avq','serv_avd']] = 0.0
        for r in range(1,19):
            self.users.loc[self.users.region_id==r,'serv_inf'] = \
                hrs_per_users_inf.loc[r]
            self.users.loc[self.users.region_id==r,'serv_avq'] = \
                hrs_per_users_avq.loc[r]
            self.users.loc[self.users.region_id==r,'cost'] = \
                self.registry.loc[r,'cah']
        self.users['serv_avd'] = self.users['needs_avd']
        for c in ['inf','avq','avd']:
            self.users['tx_serv_'+c] = 100.0*(self.users['serv_'+c]/self.users[
                'needs_'+c])
            self.users['tx_serv_'+c].clip(lower=0.0,upper=100.0,inplace=True)
        self.users['cost'] *= 1/12
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
