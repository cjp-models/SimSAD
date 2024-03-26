import pandas as pd
import numpy as np
import os
from itertools import product
from .needs import needs
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'simsad/data')
pd.options.mode.chained_assignment = None

class rpa:
    """
    Résidence privée pour aînés (RPA)

    Cette classe permet de modéliser les différents aspects des résidences privées pour aînés.

    Parameters
    ----------
    policy: object
        paramètres du scénario
    """
    def __init__(self, policy):
        self.policy = policy
        self.opt_penetrate = self.policy.rpa_penetrate
        self.penetrate_rate = self.policy.rpa_penetrate_rate
        self.adapt_rate = self.policy.rpa_adapt_rate
        return
    def load_register(self,start_yr=2019):
        """
        Fonction qui permet de charger les paramètres liés aux RPA et qui crée le registre de ceux-ci.
        Ce registre contient de l'information sur le nombre de personnes, 
        leur profil Iso-SMAF et le nombre de personnes en attente d'une place en RPA subventionnée.

        Parameters
        ----------
        start_yr: integer
            année de référence (défaut=2019)
        """
        reg = pd.read_csv(os.path.join(data_dir,'registre_rpa.csv'),
            delimiter=';',low_memory=False)
        reg = reg[reg.annee==start_yr]
        reg.set_index(['region_id'],inplace=True)
        reg.drop(labels='annee',axis=1,inplace=True)
        keep_vars = ['nb_places','nb_installations','nb_usagers_sad']
        for s in range(1,15):
            keep_vars.append('iso_smaf'+str(s))
        reg = reg[keep_vars]
        reg.rename({'nb_usagers_sad':'nb_usagers'},axis=1,inplace=True)
        # reset smaf allocation of patients
        self.registry = reg
        self.registry['nb_places_sad'] = self.registry.loc[:,'nb_usagers']
        self.days_per_year = 365
        self.nsmafs = 14
        self.registry['attente_usagers_mois'] = 0.0
        return
    def assign(self,applicants,waiting_users,region_id):
        """
        Fonction qui comptabilise dans le registre des RPA les profils Iso-SMAF des usagers, 
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
        Fonction qui permet le développement de places en RPA subventionnées par le public.
        """
        if self.opt_penetrate:
            work = self.registry.copy()
            work['cap'] = work['nb_places'] * self.penetrate_rate
            for r in range(1,19):
                row = work.loc[r,:]
                if (row['nb_places_sad']+row['attente_usagers'])<=row['cap']:
                    row['nb_places_sad'] += row['attente_usagers']
                else :
                    row['nb_places_sad'] = row['cap']
                self.registry.loc[r,'nb_places_sad'] = row['nb_places_sad']
        return
    def create_users(self, users):
        """
        Fonction qui crée le dataframe du bassin d'individus en RPA subventionnées.

        Parameters
        ----------
        users: dataframe
            Nombre d'usagers en RPA subventionnées par région, profil Iso-SMAF et groupe d'âge
        """
        self.users = users.to_frame()
        self.users.columns = ['wgt']
        self.users.loc[self.users.wgt.isna(), 'wgt'] = 0.0
        self.users.wgt.clip(lower=0.0, inplace=True)
        self.users.wgt = self.users.wgt.apply(lambda x: np.round(x)).astype('int64')
        sample_ratio = 0.25
        self.users.wgt *= sample_ratio
        self.users = self.users.reindex(self.users.index.repeat(self.users.wgt))
        self.users.wgt = 1/sample_ratio
        self.users['smaf'] = self.users.index.get_level_values(1)
        self.users['milieu'] = 'rpa'
        self.users['supplier'] = 'public'
        n = needs()
        for c in ['inf','avq','avd']:
            self.users['needs_'+c] = 0.0
        for s in range(1,15):
            self.users.loc[self.users.smaf==s,'needs_inf'] = n.inf[
                                                                 s-1]*self.days_per_year
            self.users.loc[self.users.smaf==s,'needs_avq'] = n.avq[
                                                                 s-1]*self.days_per_year
            self.users.loc[self.users.smaf==s,'needs_avd'] = n.avd[
                                                                 s-1]*self.days_per_year
        self.users['tx_serv_inf'] = 0.0
        self.users['tx_serv_avq'] = 0.0
        self.users['tx_serv_avd'] = 0.0
        self.users['wait_time'] = 0.0
        self.users['cost'] = 0.0
        self.users['any_svc'] = True
        self.users = self.users.reset_index()
        self.users['id'] = np.arange(len(self.users))
        self.users.set_index('id',inplace=True)
        return
    def update_users(self):
        """
        Fonction qui met à jours les caractéristiques des personnes dans le bassin d'individus en RPA subventionnée.
        """
        # services
        self.users[['serv_inf', 'serv_avq', 'serv_avd']] = 0.0
        # clsc
        for c in ['inf','avq','avd']:
            self.users['serv_'+c] += self.users['clsc_'+c+'_hrs']
        # pefsad
        self.users['serv_avd'] += self.users['pefsad_avd_hrs']
        self.users['cost'] = self.users['pefsad_contrib']
        # taux de service
        for c in ['inf','avq','avd']:
            self.users['tx_serv_'+c] = 100.0*(self.users['serv_'+c]/self.users[
                'needs_'+c])
            self.users['tx_serv_' + c].clip(lower=0.0, upper=100.0,
                                        inplace=True)
        self.users['cost'] *= 1 / 12
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