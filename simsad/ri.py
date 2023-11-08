import pandas as pd
import numpy as np
import os
from .needs import needs
from itertools import product
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'simsad/data')
pd.options.mode.chained_assignment = None


class ri:
    """
    Ressources intermédiaires et ressources de type familial (RI-RTF)

    Cette classe permet de modéliser les différents aspects des ressources intermédiaires 
    et ressources de type familial.

    Parameters
    ----------
    policy: object
        paramètres du scénario
    """
    def __init__(self, policy):
        self.policy = policy
        self.opt_build = self.policy.ri_build
        self.build_rate = self.policy.ri_build_rate
        self.avq_rate = self.policy.ri_avq_rate
        self.avd_rate = self.policy.ri_avd_rate
        return
    def load_register(self,start_yr=2019):
        """
        Fonction qui permet de charger les paramètres liés aux RI-RTF et qui crée le registre de ceux-ci.
        Ce registre contient des informations sur le nombre de places, le nombre d'usagers, leur profil Iso-SMAF,
        les heures de services fournis, la main d'oeuvre et les coûts.
        
        Parameters
        ----------
        start_yr: int
            année de référence (défaut=2019)
        """
        reg = pd.read_csv(os.path.join(data_dir,'registre_ri.csv'),
            delimiter=';',low_memory=False)
        reg = reg[reg.annee==start_yr]
        reg.set_index(['region_id'],inplace=True)
        reg.drop(labels='annee',axis=1,inplace=True)
        # reset smaf allocation of patients
        self.registry = reg
        # needs weights (hours per day of care by smaf, Tousignant)
        n = needs()
        self.needs_avd = n.avd
        self.needs_avq = n.avq
        self.days_per_week = 7
        self.days_per_year = 365
        self.hours_per_day = 7.5
        self.time_per_pause = 0.5
        self.nsmafs = 14
        self.weeks_per_year = (self.days_per_year/self.days_per_week)
        self.registry['attente_usagers_mois'] = 0.0

        self.registry['hours_per_etc_ri_avq'] = self.registry[
                                                 'heures_tot_trav_ri_avq']/self.registry['nb_etc_ri_avq']
        self.registry['hours_per_etc_ri_avd'] = self.registry[
                                               'heures_tot_trav_ri_avd']/self.registry['nb_etc_ri_avd']

        self.registry['etc_avq_per_user'] = self.registry[
                                                        'nb_etc_ri_avq']/self.registry['nb_usagers']
        self.registry['etc_avd_per_user'] = self.registry[
                                                        'nb_etc_ri_avd']/self.registry['nb_usagers']
        self.registry['places_per_installation'] = self.registry['nb_places']/self.registry['nb_installations']
        return
    def assign(self,applicants,waiting_users,region_id):
        """
        Fonction qui comptabilise dans le registre des RI-RTF les profils Iso-SMAF des usagers, 
        le nombre de ceux-ci, et le nombre de personnes en attente d'une place.

        Parameters
        ----------
        applicants: dataframe
            Nombre d'usagers par profil Iso-SMAF
        waiting_users: dataframe
            Nombre de personnes en attente d'une place
        region_id: int
            Numéro de la région    
        """
        self.registry.loc[region_id,['iso_smaf'+str(s) for s in range(1,15)]] = applicants
        self.registry.loc[region_id,'nb_usagers'] = np.sum(applicants)
        self.registry.loc[region_id,'attente_usagers'] = waiting_users
        return 
    def build(self):
        """
        Fonction qui permet de développer des places en RI-RTF.
        """
        if self.opt_build:
            work = self.registry[['attente_usagers', 'nb_places']].copy()
            build = self.build_rate * work['attente_usagers']
            self.registry['nb_places'] += build
            self.registry['nb_etc_ri_avq'] += self.registry[
                'etc_avq_per_user'] * build * self.avq_rate
            self.registry['nb_etc_ri_avd'] += self.registry[
                'etc_avd_per_user'] * build * self.avd_rate
        return
    def compute_supply(self):
        """
        Fonction qui calcule le nombre total d'heures de services pouvant être fournies en RI-RTF (AVQ et AVD)
        selon la main d'oeuvre disponible.
        """
        self.registry['heures_tot_trav_ri_avq'] = self.registry[
            'hours_per_etc_ri_avq'] * self.registry['nb_etc_ri_avq']
        self.registry['heures_tot_trav_ri_avd'] = self.registry[
            'hours_per_etc_ri_avd'] * self.registry['nb_etc_ri_avd']
        ## avq
        time_avq = self.registry['hours_per_etc_ri_avq'].copy()
        # take out pauses
        time_avq -= self.time_per_pause*(time_avq/self.hours_per_day)
        # blow up using number of AVQ workers
        time_avq  = time_avq * self.registry['nb_etc_ri_avq']
        # avd
        time_avd = self.registry['hours_per_etc_ri_avd'].copy()
        # take out pauses
        time_avd -= self.time_per_pause*(time_avd/self.hours_per_day)
        # blow up using number of AVD workers
        time_avd  = time_avd * self.registry['nb_etc_ri_avd']
        result = pd.concat([time_avq,time_avd],axis=1)
        result.columns = ['avq','avd']
        self.registry['supply_avq'] = result['avq']
        self.registry['supply_avd'] = result['avd']
        return result
    def compute_costs(self):
        """
        Fonction qui calcule les coûts des RI-RTF.
        """
        self.registry['cout_fixe'] = self.registry['nb_usagers'] * self.registry['cout_place_fixe']
        self.registry['cout_var'] = 0.0
        for s in range(1,15):
            self.registry['cout_var'] += (self.registry['cout_place_var'+str(s)] * self.registry['iso_smaf'+str(s)])
        self.registry['cout_total'] = self.registry['cout_fixe'] + self.registry['cout_var']
        self.registry['cout_place_total'] = self.registry['cout_total']/self.registry['nb_usagers']
        return
    def update_users(self):
        """
        Fonction qui met à jours les caractéristiques des personnes dans le bassin d'individus en RI-RTF.
        """
        # get how many hours are supplied for each domain
        hrs_per_users_avd = self.registry['heures_tot_trav_ri_avd'] / \
                            self.registry[
            'nb_usagers']
        hrs_per_users_avq = self.registry['heures_tot_trav_ri_avq'] / \
                            self.registry[
            'nb_usagers']
        self.users[['serv_inf', 'serv_avq', 'serv_avd']] = 0.0
        # services for nurses coming from CLSC
        self.users['serv_inf'] = self.users['clsc_inf_hrs']
        for r in range(1, 19):
            self.users.loc[self.users.region_id == r, 'serv_avq'] = \
                hrs_per_users_avq.loc[r]
            self.users.loc[self.users.region_id == r, 'serv_avd'] = \
                hrs_per_users_avd.loc[r]
            self.users.loc[self.users.region_id == r, 'cost'] = \
                self.registry.loc[r, 'cah'] * (1.0 +
                                               self.policy.delta_cah_ri/100.0)
        for c in ['inf', 'avq', 'avd']:
            self.users['tx_serv_' + c] = 100.0 * (
                        self.users['serv_' + c] / self.users[
                    'needs_' + c])
            self.users['tx_serv_' + c].clip(lower=0.0, upper=100.0,
                                            inplace=True)
        self.users['cost'] *= 1 / 12
        return

    def create_users(self, users):
        """
        Fonction qui crée le dataframe du bassin d'individus en RI-RTF.

        Parameters
        ----------
        users: dataframe
            Nombre d'usagers en RI-RTF par région, profil Iso-SMAF et groupe d'âge
        """
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
        self.users['milieu'] = 'ri'
        self.users['supplier'] = 'public'
        n = needs()
        for c in ['inf','avq','avd']:
            self.users['needs_'+c] = 0.0
        for s in range(1,15):
            self.users.loc[self.users.smaf==s,'needs_inf'] = n.inf[s-1]*self.days_per_year
            self.users.loc[self.users.smaf==s,'needs_avq'] = n.avq[s-1]*self.days_per_year
            self.users.loc[self.users.smaf==s,'needs_avd'] = n.avd[s-1]*self.days_per_year
        self.users['tx_serv_inf'] = 0.0
        self.users['tx_serv_avq'] = 0.0
        self.users['tx_serv_avd'] = 0.0
        self.users['wait_time'] = 0.0
        self.users['cost'] = 0.0
        self.users['any_svc'] = False
        self.users = self.users.reset_index()
        self.users['id'] = np.arange(len(self.users))
        self.users.set_index('id',inplace=True)
        return
    def reset_users(self):
        self.users = []
        return
    def collapse(self, domain = 'registry', rowvars=['region_id'],colvars=['smaf']):
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