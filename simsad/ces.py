import pandas as pd
import numpy as np
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'simsad/data')


class ces:
    """
    Chèque emploi-service (CES)

    Cette classe permet d'attribuer les services reçus dans le cadre du Chèque emploi-service.
    """
    def __init__(self):
        self.load_params()
        return
    def load_params(self, start_yr = 2021):
        """
        Fonction qui permet de charger les paramètres liés au CES.

        Parameters
        ----------
        start_yr: int
            année de référence (défaut=2021)
        """
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
        """
        Fonction qui attribue aux individus les heures de services reçues dans le cadre du CES.

        Parameters
        ----------
        users: dataframe
            bassin d'individus d'un milieu de vie donné
        
        Returns
        -------
        users: dataframe
            bassin d'individus pour un milieu de vie donné
        """
        merge_key = ['region_id','iso_smaf','gr_age']
        work = users.copy()
        work = work.merge(self.prob,left_on = merge_key,right_on = merge_key,
                   how = 'left')
        work['u'] = np.random.uniform(size=len(work))
        work['ces_any'] = (work['u']<=work['prob'])

        work = work.merge(self.hrs,left_on = merge_key,right_on = merge_key,
                   how = 'left')
        work['ces_hrs_avq'] = 0.0
        work['ces_hrs_avd'] = 0.0
        work.loc[work.ces_any,'ces_hrs_avq'] = work.loc[work.ces_any,'hrs_avq']
        work.loc[work.ces_any,'ces_hrs_avd'] = work.loc[work.ces_any,'hrs_avd']
        users[['ces_any','ces_hrs_avd','ces_hrs_avq']] = work.loc[:,['ces_any',
                                                                'ces_hrs_avd','ces_hrs_avq']]
        return users

    def calibrate(self, users, targets_by_region):
        """
        Fonction qui calibre les heures de services reçues dans le cadre du CES par rapport aux données observées.

        Parameters
        ----------
        users: dataframe
            bassin d'individus d'un milieux de vie donné
        targets_by_region: dataframe
            valeurs cibles par région
        """
        merge_key = ['region_id','iso_smaf','gr_age']
        work = users.copy()
        work = work.merge(self.prob,left_on = merge_key,right_on = merge_key,
                   how = 'left')
        users_by_region = work.groupby('region_id').apply(lambda d: (d['prob'] *
                                                                      d.wgt).sum())
        factor = targets_by_region['nb_usagers']/users_by_region

        factor[factor.isna()] = 1.0
        factor.clip(lower=0.0,upper=5.0,inplace=True)
        for r in range(1,19):
            select = self.prob.index.get_level_values(0)==r
            self.prob.loc[select,'prob'] = \
                self.prob.loc[select,'prob']*factor[r]
        work = users.copy()
        work = work.merge(self.prob,left_on = merge_key,right_on = merge_key,
                   how = 'left')
        users_by_region = work.groupby('region_id').apply(lambda d: (d['prob'] *
                                                                      d.wgt).sum())
        # hours (use updated probs)
        work = users.copy()
        work = work.merge(self.prob,left_on = merge_key,right_on = merge_key,
                   how = 'left')
        work = work.merge(self.hrs,left_on = merge_key,right_on = merge_key,
                   how = 'left')
        # get total for hours per region
        hrs_avq_by_region = work.groupby('region_id').apply(lambda d: (d[
                                                                           'prob']*d[
                                                                         'hrs_avq'] *
                                                                      d.wgt).sum())
        hrs_avd_by_region = work.groupby('region_id').apply(lambda d: (d[
                                                                           'prob']*d[
                                                                         'hrs_avd'] *
                                                                      d.wgt).sum())
        # compute factors
        factor_avq = targets_by_region['heures_tot_trav_avq']*(1.0-targets_by_region['tx_hrs_dep_avq']) / \
                     hrs_avq_by_region
        factor_avq.clip(lower=0.0,upper=5.0,inplace=True)
        factor_avq[factor_avq.isna()] = 1.0
        factor_avd = targets_by_region['heures_tot_trav_avd']*(1.0-targets_by_region['tx_hrs_dep_avd']) / \
                     hrs_avd_by_region
        factor_avd.clip(lower=0.0,upper=5.0,inplace=True)
        factor_avd[factor_avd.isna()] = 1.0
        for r in range(1,19):
            select = self.hrs.index.get_level_values(0)==r
            self.hrs.loc[select,'hrs_avq'] = self.hrs.loc[select,
            'hrs_avq']*factor_avq.loc[r]
            self.hrs.loc[select,'hrs_avd'] = self.hrs.loc[select,
            'hrs_avd']*factor_avd.loc[r]
        return

    def collapse(self, domain = 'registry', rowvars=['region_id'],colvars=['smaf']):
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


