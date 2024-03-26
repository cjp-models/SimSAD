import pandas as pd
import numpy as np
import os
from .needs import needs
from itertools import product
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'simsad/data')
pd.options.mode.chained_assignment = None


class home:
    """
    Personnes à domicile

    Cette classe permet de modéliser les personnes à domicile.
    
    Parameters
    ----------
    policy: object
        paramètres du scénario
    """
    def __init__(self, policy):
        self.policy = policy
        return
    def load_register(self):
        """
        Fonction qui permet de créer le registre des personnes à domicile. 
        Ce registre contient des informations sur le nombre de personnes et les profils Iso-SMAF de celles-ci.
        """
        self.registry = pd.DataFrame(index=range(1,19))
        self.registry[['iso_smaf_svc'+str(s) for s in range(1,15)]] = 0.0
        self.registry['nb_usagers_svc'] = 0.0
        self.registry[['iso_smaf_none'+str(s) for s in range(1,15)]] = 0.0
        self.registry['nb_usagers_none'] = 0.0
        self.registry['attente_usagers'] = 0.0
        self.days_per_year = 365
        return
    def assign(self,applicants_none, applicants_svc, waiting_users, region_id):
        """
        Fonction qui répertorie dans le registre des personnes à domicile le nombre de personnes, 
        leur profil Iso-SMAF et le nombre de personnes en attente d'une place en SAD en résidence familiale.

        Parameters
        ----------
        applicants_none: dataframe
            Nombre de personnes à domicile sans services par profil Iso-SMAF
        applicants_svc: dataframe
            Nombre de personnes à domicile avec du SAD par profil Iso-SMAF
        waiting_users: object
            Nombre de personnes en attente de SAD
        region_id: int
            numéro de région  
        """
        self.registry.loc[region_id,['iso_smaf_svc'+str(s) for s in range(1,15)]] = applicants_svc
        self.registry.loc[region_id,'nb_usagers_svc'] = np.sum(applicants_svc)
        self.registry.loc[region_id,['iso_smaf_none'+str(s) for s in range(1,15)]] = applicants_none
        self.registry.loc[region_id,'nb_usagers_none'] = np.sum(applicants_none)
        self.registry.loc[region_id,'attente_usagers'] = waiting_users
        return
    def create_users(self, users_none, users_svc):
        """
        Fonction qui crée le bassin d'individus à domicile.

        Parameters
        ----------
        applicants_none: dataframe
            Nombre de personnes à domicile sans services par région, profil Iso-SMAF et groupe d'âge
        applicants_svc: dataframe
            Nombre de personnes à domicile avec du SAD par région, profil Iso-SMAF et groupe d'âge
        """
        # users with services
        users_svc = users_svc.to_frame()
        users_svc.columns = ['wgt']
        users_svc['any_svc'] = True
        users_svc.loc[users_svc.wgt.isna(), 'wgt'] = 0.0
        users_svc.wgt.clip(lower=0.0, inplace=True)
        users_svc = users_svc.reset_index()
        users_svc.set_index(['region_id','iso_smaf','gr_age','any_svc'], inplace = True)
        # users without services
        users_none = users_none.to_frame()
        users_none.columns = ['wgt']
        users_none['any_svc'] = False
        users_none.loc[users_none.wgt.isna(), 'wgt'] = 0.0
        users_none.wgt.clip(lower=0.0, inplace=True)
        users_none = users_none.reset_index()
        users_none.set_index(['region_id','iso_smaf','gr_age','any_svc'], inplace = True)
        self.users = pd.concat([users_svc,users_none],axis=0)
        sample_ratio = 0.1
        self.users.wgt *= sample_ratio
        self.users.wgt = self.users.wgt.apply(lambda x: np.round(x)).astype('int64')
        self.users = self.users.reindex(self.users.index.repeat(self.users.wgt))
        self.users.wgt = 1/sample_ratio
        self.users['smaf'] = self.users.index.get_level_values(1)
        self.users['milieu'] = 'home'
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
        self.users = self.users.reset_index()
        self.users['id'] = np.arange(len(self.users))
        self.users.set_index('id',inplace=True)
        return

    def update_users(self):
        """
        Fonction qui met à jour les caractéristiques du bassin d'individus à domicile.
        """
        # services
        self.users[['serv_inf', 'serv_avq', 'serv_avd']] = 0.0
        # clsc
        for c in ['inf','avq','avd']:
            self.users['serv_'+c] += self.users['clsc_'+c+'_hrs']
        # pefsad
        self.users['serv_avd'] += self.users['pefsad_avd_hrs']
        self.users['cost'] = self.users['pefsad_contrib']
        # ces
        self.users['serv_avq'] +=  self.users['ces_hrs_avq']
        self.users['serv_avd'] +=  self.users['ces_hrs_avd']
        # taux de service
        for c in ['inf','avq','avd']:
            self.users['tx_serv_'+c] = 100.0*(self.users['serv_'+c]/self.users[
                'needs_'+c])
            self.users['tx_serv_' + c].clip(lower=0.0, upper=100.0,
                                        inplace=True)
        self.users['cost'] *= 1/12
        return
    def reset_users(self):
        """
        Fonction qui réinitialise le bassin d'individus à domicile.
        """
        self.users = []
        return
    def collapse(self, domain = 'registry', rowvars=['region_id'],colvars=['smaf']):
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
