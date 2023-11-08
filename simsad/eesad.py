import pandas as pd
import numpy as np
from itertools import product
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'simsad/data')
from numba import njit, float64, int64, boolean, prange
from numba.types import Tuple
from .needs import needs
from itertools import product



class eesad:
    """
    Entreprises d'économie sociale en aide à domicile (EÉSAD)

    Cette classe permet de modéliser les services offerts en soutien à l'autonomie par les entreprises d'économie sociale en aide à domicile.
    
    Parameters
    ----------
    policy: object
        paramètres du scénario
    """
    def __init__(self, policy):
        self.load_registry()
        self.policy = policy
        return
    def load_registry(self, start_yr = 2019):
        """
        Fonction qui permet de créer le registre des EÉSAD. Ce registre contient des informations concernant 
        la main d'oeuvre et les coûts.

        Parameters
        ----------
        start_yr: integer
            année de référence (défaut=2019)
        """
        self.registry = pd.read_csv(os.path.join(data_dir, 'registre_eesad.csv'),
                                        delimiter=';', low_memory=False)
        self.registry = self.registry[self.registry.annee == start_yr]
        self.registry.set_index('region_id', inplace=True)
        self.registry.drop(labels='annee',axis=1,inplace=True)
        self.registry['hrs_per_etc'] = self.registry['heures_tot_trav_avd']/self.registry['nb_etc_avd']
        tups = list(product(*[np.arange(1,19),np.arange(1,15)]))
        itups = pd.MultiIndex.from_tuples(tups)
        itups.names = ['region_id','smaf']
        self.count = pd.DataFrame(index=itups,columns =['users','hrs'],dtype='float64')
        self.count.loc[:,:] = 0.0
        self.days_per_year = 365
        return
    def assign(self, users_home, users_rpa):
        """
        Fonction qui calcule la contribution des usagers des EÉSAD dans les bassins de d'individus à domicile ou en RPA,
        et qui comptabilise le nombre d'usagers et le nombre total d'heures de services fournis (en AVD) par région et profil Iso-SMAF.

        Parameters
        ----------
        users_home: dataframe
            Bassin d'individus à domicile
        users_rpa: dataframe
            Bassin d'individus en RPA
        
        Returns
        -------
        users_home: dataframe
            Bassin d'individus à domicile
        users_home: dataframe
            Bassin d'individus en RPA  
        """
        users_home['pefsad_contrib'] = 0.0
        users_rpa['pefsad_contrib'] = 0.0
        for r in range(1,19):
            users_home.loc[users_home.region_id==r,'pefsad_contrib'] = \
                self.registry.loc[r,'contribution_usager'] * users_home.loc[
                    users_home.region_id==r,'pefsad_avd_hrs']
            users_rpa.loc[users_rpa.region_id==r,'pefsad_contrib'] = \
                self.registry.loc[r,'contribution_usager'] * users_rpa.loc[
                    users_rpa.region_id==r,'pefsad_avd_hrs']
            for s in range(1,15):
                select = (users_home['region_id']==r) & (users_home['smaf']==s) & (users_home['pefsad_avd_any'])
                self.count.loc[(r,s),'users'] = users_home.loc[select,'wgt'].sum()
                self.count.loc[(r,s),'hrs'] = users_home.loc[select,['wgt','pefsad_avd_hrs']].prod(axis=1).sum()
                select = (users_rpa['region_id']==r) & (users_rpa['smaf']==s) & (users_rpa['pefsad_avd_any'])
                self.count.loc[(r,s),'users'] += users_rpa.loc[select,'wgt'].sum()
                self.count.loc[(r,s),'hrs'] += users_rpa.loc[select,['wgt','pefsad_avd_hrs']].prod(axis=1).sum()
        return users_home, users_rpa
    def compute_supply(self):
        """
        Fonction qui calcule le nombre total d'heures de services pouvant être fournies en EÉSAD (AVD)
        selon la main d'oeuvre disponible.
        """
        self.registry['supply_avd'] = self.registry['nb_etc_avd'] * self.registry['hrs_per_etc']
        self.registry['supply_avd'] *= (1.0 - self.registry['tx_hrs_dep_avd'] - self.registry['tx_hrs_admin_avd'])
        return
    def cap(self, users_home, users_rpa):
        """
        Fonction qui ajuste les heures de services fournis au niveau individuel (AVD) selon la main d'oeuvre disponible 
        et qui comptabilise les besoins supplémentaires en main d'oeuvre pour les EÉSAD.

        Parameters
        ----------
        users_home: dataframe
            Bassin d'individus à domicile
        users_home: dataframe
            Bassin d'individus en RPA  

        Returns
        -------
        users_home: dataframe
            Bassin d'individus à domicile
        users_home: dataframe
            Bassin d'individus en RPA    
        """
        hrs_home = users_home.groupby('region_id').apply(lambda d: (d['pefsad_avd_hrs']*d['wgt']).sum())
        hrs_rpa = users_rpa.groupby('region_id').apply(lambda d: (d['pefsad_avd_hrs']*d['wgt']).sum())
        hrs_tot = hrs_home + hrs_rpa
        hrs_tot[hrs_tot.isna()] = 0.0
        factor = self.registry['supply_avd']/hrs_tot
        factor.clip(upper=1.0, inplace=True)
        factor[factor.isna()] = 1.0
        excess = hrs_tot - self.registry['supply_avd']
        excess.clip(lower=0.0, inplace=True)
        excess[excess.isna()] = 0.0
        indirect = (1.0 - self.registry['tx_hrs_dep_avd'] - self.registry['tx_hrs_admin_avd'])
        excess = excess / indirect
        excess = excess / self.registry['hrs_per_etc']
        self.registry['worker_needs'] = excess
        self.registry.loc[self.registry.worker_needs.isna(),'worker_needs'] =\
            0.0
        for r in range(1,19):
            users_home.loc[users_home.region_id==r,'pefsad_avd_hrs'] *= factor[r]
            users_rpa.loc[users_rpa.region_id==r,'pefsad_avd_hrs'] *= factor[r]
            users_home.loc[users_home.region_id==r,'pefsad_contrib'] = \
                self.registry.loc[r,'contribution_usager'] * users_home.loc[
                    users_home.region_id==r,'pefsad_avd_hrs']
            users_rpa.loc[users_rpa.region_id==r,'pefsad_contrib'] = \
                self.registry.loc[r,'contribution_usager'] * users_rpa.loc[
                    users_rpa.region_id==r,'pefsad_avd_hrs']
        return users_home, users_rpa
    def compute_costs(self):
        """
        Fonction qui calcule les coûts des EÉSAD.
        """
        self.registry['cout_fixe'] = 0.0
        self.registry['cout_var'] = self.registry['sal_avd'] * self.registry['nb_etc_avd'] * self.registry['hrs_per_etc']
        self.registry['cout_total'] = self.registry['cout_fixe'] + self.registry['cout_var']
        return
    def workforce(self,before_base_yr=False):
        """
        Fonction qui ajuste le nombre d'ETC en EÉSAD selon les besoins supplémentaires en main d'oeuvre et
        le taux d'ajustement de celle-ci.

        Parameters
        ----------
        before_base_yr: boolean
            True si l'année en cours de la simulation est inférieure 
            à l'année de départ de la comptabilisation des résultats 
        """
        if before_base_yr:
            self.policy.eesad_avd_rate = 1.0
        self.registry['nb_etc_avd'] += self.policy.eesad_avd_rate * \
                                           self.registry['worker_needs']
        return
    def collapse(self, domain = 'registry', rowvars=['region_id'],colvars=['needs_inf']):
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

