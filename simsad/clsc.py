import pandas as pd
import numpy as np
from itertools import product
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'simsad/data')
from numba import njit, float64, int64, boolean, prange
from numba.types import Tuple
from .needs import needs
from itertools import product

class clsc:
    """
    Centres locaux de services communautaires (CLSC)

    Cette classe permet de modéliser les services offerts en soutien à l'autonomie par les Centres locaux 
    de services communautaires.
    
    Parameters
    ----------
    policy: object
        paramètres du scénario
    """
    def __init__(self, policy):
        self.care_types = ['inf','avq','avd']
        self.purchase_prive = policy.purchase_prive
        self.purchase_eesad = policy.purchase_eesad
        self.policy = policy
        self.clsc_inf_rate = policy.clsc_inf_rate
        self.clsc_avq_rate = policy.clsc_avq_rate
        self.clsc_avd_rate = policy.clsc_avd_rate
        self.load_params()
        self.load_registry()
        return
    def load_registry(self, start_yr = 2019):
        """
        Fonction qui permet de créer le registre des CSLC. Ce registre contient des informations concernant 
        la main d'oeuvre, les heures de services fournis et les coûts.

        Parameters
        ----------
        start_yr: int
            année de référence (défaut=2019)
        """
        self.registry = pd.read_csv(os.path.join(data_dir, 'registre_clsc.csv'),
                                        delimiter=';', low_memory=False)
        self.registry = self.registry[self.registry.annee==start_yr]
        self.registry.set_index('region_id',inplace=True)
        self.registry.drop(labels='annee',axis=1,inplace=True)
        self.days_per_year = 365
        self.registry['nb_etc_inf'] = 0
        self.registry['nb_etc_avq'] = 0
        self.registry['nb_etc_avd'] = 0
        for m in ['home','rpa','ri']:
            if m!='ri':
                for c in self.care_types:
                    tag = c+'_'+m
                    self.registry['hrs_etc_' + tag] = self.registry[
                        'heures_tot_trav_'+tag]/self.registry[
                        'nb_etc_'+tag]
                    self.registry['nb_etc_'+c] += self.registry['nb_etc_'+tag]
            else :
                tag = 'inf_' + m
                self.registry['hrs_etc_' + tag] = self.registry[
                                                       'heures_tot_trav_' + tag] / \
                                                   self.registry[
                                                       'nb_etc_' + tag]
                self.registry['nb_etc_inf'] += self.registry['nb_etc_' + tag]

        self.registry['cout_hr_achete_eesad_avq'] = self.registry.loc[:,'cout_hr_achete_prive_avq']
        self.purchase_home = pd.DataFrame(index=['inf','avq','avd'],columns=['prive','eesad'])
        self.purchase_home.loc[:,:] = False
        if self.purchase_eesad:
            self.purchase_home.loc['avq','eesad'] = True
            self.purchase_home.loc['avd','eesad'] = True
        else :
            self.registry['hrs_sa_avq_eesad_home'] = 0.0
            self.registry['hrs_sa_avd_eesad_home'] = 0.0

        if self.purchase_prive:
            self.purchase_home.loc['inf','prive'] = True
            self.purchase_home.loc['avq','prive'] = True
            self.purchase_home.loc['avd','prive'] = True
        else :
            self.registry['hrs_sa_inf_prive_home'] = 0.0
            self.registry['hrs_sa_avq_prive_home'] = 0.0
            self.registry['hrs_sa_avd_prive_home'] = 0.0


        self.purchase_rpa = pd.DataFrame(index=['inf','avq','avd'],columns=['prive','eesad'])
        self.purchase_rpa.loc[:,:] = False
        if self.purchase_eesad:
            self.purchase_rpa.loc['avq','eesad'] = True
            self.purchase_rpa.loc['avd','eesad'] = True
        else :
            self.registry['hrs_sa_avq_eesad_rpa'] = 0.0
            self.registry['hrs_sa_avd_eesad_rpa'] = 0.0

        if self.purchase_prive:
            self.purchase_rpa.loc['avq','prive'] = True
            self.purchase_rpa.loc['avd','prive'] = True
        else :
            self.registry['hrs_sa_avq_prive_rpa'] = 0.0
            self.registry['hrs_sa_avd_prive_rpa'] = 0.0

        return
    def load_params(self):
        """
        Fonction qui permet de charger les paramètres liés aux heures de services fournis par les CSLC.
        """
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
    def assign(self, users, milieu, policy, yr):
        """
        Fonction qui détermine les usagers des CLSC et qui leur attribue des heures de services fournis 
        (soins infirmier, AVQ, AVD).

        Parameters
        ----------
        users: dataframe
            Bassin d'individus d'un milieu de vie donné
        milieu: str
            Nom du milieu de vie
        policy: object
            Paramètres du scénario
        yr: int
            Année en cours de la simulation  

        Returns
        -------
        users: dataframe
           Bassin d'individus d'un milieu de vie donné   
        """
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
        # for those without services, set choice to 0 (no services)
        users.loc[users.choice.isna(),'choice'] = 0
        # adjust services to reach target service rate
        if milieu=='home':
            if yr>=2023:
                n = needs()
                needs_cap ={'inf': n.inf[9]*365,'avq':n.avq[9]*365,'avd':n.avd[9]*365}
                for c in self.care_types:
                    delta = getattr(policy,'delta_'+c+'_rate')/100.0
                    smafs = [s for s in range(4,15)]
                    cond = users['iso_smaf'].isin(smafs)
                    users.loc[cond, 'clsc_'+c+'_hrs'] += delta * users.loc[cond, 'needs_'+c].clip(upper=needs_cap[c])
 
        if milieu!='ri':
            counts_any_svc = users.groupby(['region_id']).apply(
                lambda d: (d['any_svc']*d['wgt']).sum())
            self.registry['nb_usagers_'+milieu] = counts_any_svc
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
        return users
    def compute_supply(self, yr):
        """
        Fonction qui comptabilise les heures de services pouvant être fournies en CLSC (soins infirmiers et AVQ) 
        selon la main d'oeuvre disponible.

        Parameters
        ----------
        yr: int
            Année en cours de la simulation 
        """
        # hours supplied by CLSC
        for m in ['home','rpa']:
            for c in self.care_types:
                tag = c+'_'+m
                self.registry['supply_'+tag] = self.registry['nb_etc_'+tag] * \
                                              self.registry['hrs_etc_'+tag]
                self.registry['supply_'+tag] *= (1.0-self.registry[
                    'tx_hrs_dep_'+c] - self.registry[
                    'tx_hrs_admin_'+c])
        # if policy yr, outsource avq and avq
        if yr==2023:
            for m in ['home', 'rpa']:
                for c in ['avq','avd']:
                    tag = c + '_' + m
                    supply_hrs = self.registry['supply_'+tag].copy()
                    indirect = (1.0 - self.registry[
                        'tx_hrs_dep_' + c] - self.registry['tx_hrs_admin_' +
                                                           c]) * self.registry['hrs_etc_' + tag]
                    for s in ['eesad', 'prive']:
                        rate = getattr(self.policy,'clsc_shift_' + c + '_' + s)
                        self.registry['hrs_sa_' + c + '_' + s + '_'+m] += \
                            rate*supply_hrs
                        self.registry['supply_' + tag] = self.registry[
                            'supply_' + tag] - rate*supply_hrs
                        self.registry['nb_etc_' + tag] = self.registry[
                            'nb_etc_' + tag] - rate*supply_hrs/indirect
                    self.registry['supply_' + tag].clip(lower=0.0,inplace=True)
                    self.registry['nb_etc_' + tag].clip(lower=0.0,inplace=True)
        tag = 'inf_ri'
        self.registry['supply_'+tag] = self.registry['nb_etc_'+tag] * \
                                              self.registry['hrs_etc_'+tag]
        self.registry['supply_'+tag] *= (1.0-self.registry[
                    'tx_hrs_dep_inf'] - self.registry[
                    'tx_hrs_admin_inf'])

        # add hours purchased from EESAD and PRIVE for home care
        for c in self.care_types:
            for s in ['prive','eesad']:
                if self.purchase_home.loc[c,s]:
                    self.registry['supply_'+c+'_home'] += self.registry['hrs_sa_'+c+'_'+s+'_home']

        # add hours purchased add hours purchased from EESAD and PRIVE for RPA
        for c in self.care_types:
            for s in ['prive','eesad']:
                if self.purchase_rpa.loc[c,s]:
                    self.registry['supply_'+c+'_rpa'] += self.registry['hrs_sa_'+c+'_'+s+'_rpa']

        # recompile number of workers
        for c in self.care_types:
            self.registry['nb_etc_'+c] = 0.0
            for m in ['home','rpa','ri']:
                tag = c + '_' + m
                if m=='ri':
                    if c=='inf':
                        self.registry['nb_etc_'+c] += self.registry[
                            'nb_etc_'+tag]
                else :
                    self.registry['nb_etc_' + c] += self.registry[
                        'nb_etc_' + tag]

        # add up to get overall supply
        self.registry['supply_inf'] = self.registry[['supply_inf_'+c for c in
                                                    ['home','rpa','ri']
                                                    ]].sum(axis=1)
        self.registry['supply_avq'] = self.registry[['supply_avq_'+c for c in
                                                    ['home','rpa']
                                                    ]].sum(axis=1)
        self.registry['supply_avd'] = self.registry[['supply_avd_'+c for c in
                                                    ['home','rpa']
                                                    ]].sum(axis=1)
        colvars = []
        for m in ['home','rpa']:
            for c in self.care_types:
                tag = c + '_' + m
                colvars.append(tag)
        colvars.append('inf_ri')
 
        self.worker_needs = pd.DataFrame(index=np.arange(1,19),
                                         columns=colvars,dtype='float64')
        self.worker_needs.loc[:,:] = 0.0

        self.registry[['worker_needs_inf','worker_needs_avq','worker_needs_avd']] = 0.0
        return

    def cap(self, users, milieu):
        """
        Fonction qui ajuste les heures de services fournis au niveau individuel (soins infirmier, AVQ, AVD) 
        selon la main d'oeuvre disponible et qui comptabilise les besoins supplémentaires en main d'oeuvre.

        Parameters
        ----------
        users: dataframe
            Bassin d'individus d'un milieu de vie donné
        milieu: str
            Nom du milieu de vie

        Returns
        -------
        users: dataframe
           Bassin d'individus d'un milieu de vie donné   
        """
        hrs_free = pd.pivot_table(data=self.count, index=['region_id'],
                                  columns=['milieux','svc'], values='hrs',
                               aggfunc='sum')
        if milieu!='ri':
            for c in self.care_types:
                tag = c+'_'+milieu
                factor = self.registry['supply_'+tag]/hrs_free.loc[:,(milieu,c)]
                factor.clip(upper=1.0,inplace=True)
                excess = hrs_free.loc[:,(milieu,c)] - self.registry['supply_' + tag]
                excess.clip(lower=0.0,inplace=True)
                if c in ['avq','avd']:
                    excess_base = excess.copy()
                    for s in ['eesad','prive']:
                        shifting_rate = getattr(self.policy,'clsc_shift_'+c+'_'+s)
                        adjusment_rate = getattr(self.policy,s+'_'+c+'_rate')
                        self.registry['add_hrs_sa_' + c + '_' + s + '_' +
                                      milieu] = \
                            adjusment_rate * shifting_rate * excess_base
                        excess -= adjusment_rate*shifting_rate*excess_base
                indirect = (1.0 - self.registry[
                        'tx_hrs_dep_' + c] - self.registry['tx_hrs_admin_' + c])
                excess = excess / indirect
                excess = excess / self.registry['hrs_etc_' + tag]
                self.worker_needs.loc[:,tag] = excess
                tag2 = 'worker_needs_'+c
                self.registry.loc[:,tag2] += excess
                for r in range(1,19):
                    users.loc[users.region_id==r, 'clsc_'+c+'_hrs'] *= factor[r]
        else :
            tag = 'inf_ri'
            c = 'inf'
            factor = self.registry['supply_' + tag] / hrs_free.loc[:,('ri',
                                                                      'inf')]
            factor.clip(upper=1.0, inplace=True)
            excess = hrs_free.loc[:,('ri','inf')] - self.registry['supply_' + tag]
            excess.clip(lower=0.0, inplace=True)
            indirect = (1.0 - self.registry[
                'tx_hrs_dep_' + c] - self.registry['tx_hrs_admin_' + c])
            excess = excess / indirect
            excess = excess / self.registry['hrs_etc_' + tag]
            self.worker_needs.loc[:, tag] = excess
            tag2 = 'worker_needs_'+c
            self.registry.loc[:,tag2] += excess
            for r in range(1, 19):
                users.loc[users.region_id == r, 'clsc_inf_hrs'] *= factor[r]
        return users
    def compute_costs(self):
        """
        Fonction qui calcule les coûts des CLSC.
        """
        # cout fixe par usagers
        self.registry['cout_fixe'] = self.registry['cout_residuel']*(self.registry['nb_usagers_home']+self.registry['nb_usagers_rpa'])
        # cout salaire
        self.registry['cout_var'] = 0.0
        for m in ['home','rpa']:
            for c in self.care_types:
                tag = c+'_'+m
                self.registry['cout_var'] += self.registry['nb_etc_'+tag] * \
                                              self.registry['hrs_etc_'+tag] * self.registry['sal_'+c]
        tag = 'inf_ri'
        self.registry['cout_var'] += self.registry['nb_etc_'+tag] * \
                                              self.registry['hrs_etc_'+tag] * self.registry['sal_inf']
        # services acheté
        self.registry['cout_achete'] = 0.0
        for c in self.care_types:
            for s in ['prive','eesad']:
                if self.purchase_home.loc[c,s]:
                    self.registry['cout_achete'] += self.registry['hrs_sa_'+c+'_'+s+'_home'] * self.registry['cout_hr_achete_'+s+'_'+c]
        for c in self.care_types:
            for s in ['prive','eesad']:
                if self.purchase_rpa.loc[c,s]:
                    self.registry['cout_achete'] += self.registry['hrs_sa_'+c+'_'+s+'_rpa'] * self.registry['cout_hr_achete_'+s+'_'+c]

        self.registry['cout_total'] = self.registry['cout_fixe'] + self.registry['cout_var'] + self.registry['cout_achete']
        return
    def workforce(self,before_base_yr=False):
        """
        Fonction qui ajuste le nombre d'ETC en CLSC selon les besoins supplémentaires en main d'oeuvre et
        le taux d'ajustement de celle-ci.

        Parameters
        ----------
        before_base_yr: boolean
            True si l'année en cours de la simulation est inférieure 
            à l'année de départ de la comtabilisation des résultats 
        """
        self.registry['nb_etc_inf'] = 0
        self.registry['nb_etc_avq'] = 0
        self.registry['nb_etc_avd'] = 0
        for m in ['home','rpa']:
            for c in self.care_types:
                tag = c+'_'+m
                if before_base_yr:
                    attr = 1.0
                else:
                    attr = getattr(self,'clsc_'+c+'_rate')

                shift = 0.0
                if before_base_yr==False:
                    if c!='inf':
                        shift += min((getattr(self.policy,'clsc_shift_' + c + '_eesad') \
                                + getattr(self.policy,'clsc_shift_' + c + '_prive')),1)
                
                self.registry['nb_etc_'+tag] += \
                    attr * (1-shift) * self.worker_needs.loc[:,tag]
                self.registry['nb_etc_'+c] += self.registry['nb_etc_'+tag]
                
                if before_base_yr==False:
                    if c in ['avq','avd']:
                        for s in ['eesad','prive']:
                            tag_sa = 'hrs_sa_' + c + '_' + s + '_' + m
                            self.registry[tag_sa] += self.registry['add_'+tag_sa]
        tag = 'inf_ri'
        self.registry['nb_etc_' + tag] += \
            self.clsc_inf_rate * self.worker_needs.loc[:, tag]
        self.registry['nb_etc_inf'] += self.registry['nb_etc_' + tag]
        return

    def collapse_users(self,rowvars=['region_id','svc'],colvars=['iso_smaf']):
        table = pd.pivot_table(self.count.stack().to_frame(),index=rowvars,columns=colvars,aggfunc='sum')
        if colvars!=[]:
            table.columns = [x[1] for x in table.columns]
        return table
    def collapse(self, domain = 'registry', rowvars=['region_id'],colvars=['supply_inf']):
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


@njit((int64[:])(float64[:,:]), cache=True, parallel=True)
def draw_multinomial(prob):
        n, m = prob.shape
        result = np.zeros(n,dtype='int64')
        u = np.random.uniform(low=0.0,high=1.0,size=n)
        for i in prange(n):
            cp = 0.0
            for j  in range(m):
                cp += prob[i,j]
                if u[i]<=cp:
                    result[i] = j
                    break
        return result
