import pandas as pd
import numpy as np
import os
from itertools import product
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'SimSAD/data')
pd.options.mode.chained_assignment = None

class isq:
    """
    Démographie

    Cette classe permet de modéliser la démographie.

    Parameters
    ----------
    region: str
        unité géographique des projections démographiques (défaut='Québec')
    pop_fichier: str
        nom du fichier contenant les projections démographiques (défaut='pop-isq.csv')
    start_yr: int
        année de départ de la simulation (défaut=2020)
    stop_yr: int
        année de fin de la simulation (défaut=2040)   
    """
    def __init__(self,region='Québec',pop_fichier='pop-isq.csv',start_yr=2020,stop_yr=2040):
        pop = pd.read_csv(os.path.join(data_dir,pop_fichier),delimiter=';',low_memory=False)
        pop = pop[pop.sexe=='Total']
        pop = pop[pop.geo==region]
        pop.annee = pop.annee.astype('int64')
        pop = pop[pop.annee<=stop_yr]
        pop = pop[pop.annee>=start_yr]
        for c in range(91):
            pop[str(c)] = pop[str(c)].astype('int64')
        pop.drop(labels=['geo','sexe'],axis=1,inplace=True)
        self.regions = dict(pop.groupby('region_id').first()['region'])
        pop.drop(labels=['region'],axis=1,inplace=True)
        pop.set_index(['region_id','annee'],inplace=True)
        cols = []
        for c in range(91):
            cols.append(c)
        pop.columns = cols
        pop.columns.name = 'age'
        self.nregions = len(self.regions)
        self.last_region = self.nregions+1
        self.annees = pop.index.get_level_values(1).unique().to_list()
        self.count_isq = pop
        self.count = self.count_isq.loc[self.count_isq.index.get_level_values(1)==start_yr,:].droplevel(1)
        self.nannees = len(self.annees)
        return
    def evaluate(self,yr):
        """
        Fonction qui calcule le nombre de personnes par région et âge durant l'année de simulation en cours.

        yr: int
            année en cours de la simulation    
        """
        self.count = self.count_isq.loc[self.count_isq.index.get_level_values(1)==yr,:].droplevel(1)
        return
    def collapse(self,domain = 'count', rowvars=['region_id'],colvars=['age']):
        t = getattr(self, domain)
        table = pd.pivot_table(t.stack().to_frame(),index=rowvars,columns=colvars,aggfunc='sum')
        if colvars!=[]:
            table.columns = [x[1] for x in table.columns]
        return table

class gps:
    """
    Groupes de profil de santé (GPS).

    Cette classe permet de modéliser les groupes de profils de santé.

    Parameters
    ----------
    nregions: int
        nombre de régions
    ages: int
        liste des tranches d'âge considérées dans la modélisation
    ngps: int
        nombre de Groupes de profils de santé (défaut=16)  
    """
    def __init__(self,nregions, ages, ngps=16):
        self.last_region = nregions + 1
        self.regions = np.arange(1,nregions+1)
        self.nregions = len(self.regions)
        self.profiles = [p for p in range(1, ngps+1)]
        id = pd.MultiIndex.from_tuples(list(product(*[self.regions,self.profiles])))
        frame = pd.DataFrame(index=id,columns=ages)
        frame.index.names = ['region_id','gps']
        self.count = frame
        self.ngps = ngps
        self.count.columns.name = 'age'
        return
    def load(self,set_yr=2016):
        """
        Fonction qui permet de charger les paramètres liés aux GPS.

        Parameters
        ----------
        start_yr: int
            année de référence (défaut=2016)
        """
        self.shares = pd.read_csv(os.path.join(data_dir,'gps.csv'),delimiter=',',low_memory=False)
        # drop all of Quebec (keep regions), keep reference yr
        self.shares = self.shares[self.shares['region_id']!=99]
        self.shares = self.shares[self.shares.annee==set_yr]
        self.shares.drop(labels=['annee'],inplace=True,axis=1)
        self.shares.prob = self.shares.prob.astype('float64')
        self.shares.age = self.shares.age.astype('int64')
        self.shares.set_index(['region_id','gps','age'],inplace=True)
        self.shares = self.shares.unstack()
        self.shares.columns = [x for x in range(18,91)]
        # the less than 18 not evaluated in GPS
        young = [x for x in range(18)]
        self.shares[young] = 0.0
        self.shares.loc[self.shares.index.get_level_values(1)==16,young] = 1.0
        # deal with missing GPS
        for x in range(1,self.last_region):
            self.shares.loc[(x,4),:] = 0.0
        for x in range(1,self.last_region):
            self.shares.loc[(x,14),:] = 0.0
        self.shares.sort_index(inplace=True)
        # put Nans to zero
        self.shares[self.shares.isna()] = 0.0
        # re-order
        self.shares = self.shares[[x for x in range(91)]]
        return
    def evaluate(self, pop, yr):
        """
        Fonction qui calcule le nombre de personnes par GPS et région durant l'année de simulation en cours.

        pop: objet
            objet contenant les informations démographiques    
        """
        self.count = pop.count * self.shares
        return
    def collapse(self,domain = 'count', rowvars=['region_id'],colvars=['gps']):
        t = getattr(self,domain)
        table = pd.pivot_table(t.stack().to_frame(),index=rowvars,columns=colvars,aggfunc='sum')
        if colvars!=[]:
            table.columns = [x[1] for x in table.columns]
        return table

class smaf:
    """
    Profils Iso-SMAF.

    Cette classe permet de modéliser les profils Iso-SMAF.

    Parameters
    ----------
    nregions: int
        nombre de régions
    ages: int
        liste des tranches d'âge considérées dans la modélisation
    ngps: int
        nombre de groupes de profils de santé (défaut=16)
    nsmaf: int
        nombre de profils Iso-SMAF (défaut=14)    
    """
    def __init__(self, nregions, ages, ngps = 16, nsmaf = 14):
        self.last_region = nregions+1
        self.regions = np.arange(1,self.last_region)
        self.nregions = len(self.regions)
        self.nsmaf = nsmaf
        self.last_smaf = self.nsmaf + 1
        self.ngps = ngps 
        self.last_gps = self.ngps + 1
        # include a zero smaf
        self.smaf_profiles = np.arange(1,self.last_smaf)
        self.gps_profiles = [p for p in range(1, self.last_gps)]
        # evaluation frame
        id_eval = pd.MultiIndex.from_tuples(list(product(*[self.regions,self.gps_profiles])))
        frame_eval = pd.DataFrame(index=id_eval,columns=ages)
        frame_eval.index.names = ['region_id','gps']
        self.count_eval = frame_eval
        self.count_eval.columns.name = 'age'
        # smaf frame
        id_smaf = pd.MultiIndex.from_tuples(list(product(*[self.regions,self.gps_profiles,self.smaf_profiles])))
        frame_smaf = pd.DataFrame(index=id_smaf,columns=ages)
        frame_smaf.index.names = ['region_id','gps','smaf']
        self.count_smaf = frame_smaf
        self.ngps = ngps
        self.nsmaf = nsmaf
        self.count_smaf.columns.name = 'age'
        return
    def load(self,set_yr=2015):
        """
        Fonction qui permet de charger les paramètres liés aux profils Iso-SMAF.

        Parameters
        ----------
        start_yr: integer
            année de référence (défaut=2015)
        """
        # probability of evaluation
        self.prob = pd.read_csv(os.path.join(data_dir,'prob_eval.csv'),delimiter=';',low_memory=False)
        self.prob = self.prob[self.prob['region_id']!=99]
        self.prob = self.prob[self.prob['annee']==set_yr]
        self.prob.drop(labels='annee',axis=1,inplace=True)
        self.prob.set_index(['region_id','gps','age'],inplace=True)
        self.prob = self.prob.unstack()
        self.prob.columns = [x for x in range(18,91)]
        self.prob[[x for x in range(18)]] = 0.0
        self.prob = self.prob[[x for x in range(91)]]
        for x in range(1,self.last_region):
            self.prob.loc[(x,4),:] = 0.0
        for x in range(1,self.last_region):
            self.prob.loc[(x,14),:] = 0.0
        self.prob.sort_index(inplace=True)
        self.prob.columns.name = 'age'
        self.shares = pd.read_csv(os.path.join(data_dir,'prob_iso-smaf.csv'),delimiter=';',low_memory=False)
        self.shares.columns = ['region_id','age','gps','annee','smaf','prob']
        self.shares = self.shares[self.shares['region_id']!=99]
        self.shares = self.shares.loc[self.shares['annee']==set_yr,:]
        self.shares.drop(['annee'],axis=1,inplace=True)
        self.shares.set_index(['region_id','gps','smaf','age'],inplace=True)
        self.shares = self.shares.unstack()
        self.shares.columns = [x for x in range(18,91)]
        self.shares[[x for x in range(18)]] = 0.0
        self.shares = self.shares[[x for x in range(91)]]
        for x in range(1,self.last_region):
            for y in range(1,self.last_smaf):
                self.shares.loc[(x,4,y),:] = 0.0
        for x in range(1,self.last_region):
            for y in range(1,self.last_smaf):
                self.shares.loc[(x,14,y),:] = 0.0
        self.shares.sort_index(inplace=True)
        self.shares.columns.name = 'age'
        return
    def evaluate(self, grouper,gps_target=[],rate=0.0):
        """
        Fonction qui calcule le nombre de personnes par profil Iso-SMAF, GPS, région et âge 
        durant l'année de simulation en cours

        grouper: objet
            objet contenant les informations sur les GPS
        gps_target: series
            liste de GPS pour lesquels la probabilité d'être évalué pour un profil Iso-SMAF sera augmentée 
        rate: float
            taux de croissance de la probabilité d'être évalué pour un profil Iso-SMAF    
        """
        # allow for changes in prob eval
        prob = self.prob.copy()
        if gps_target:
            for g in gps_target:
                prob.loc[prob.index.get_level_values(1)==g,:] *= (1.0 + rate)
            prob = np.where(prob>1.0,1.0,prob)
        # first decide who gets evaluated
        self.count_eval = grouper.count * prob
        # from those assign to smaf
        self.count_smaf = self.shares * self.count_eval
        return
    def collapse(self, domain = 'count_smaf', rowvars=['region_id'],colvars=['smaf']):
        t = getattr(self,domain)
        table = pd.pivot_table(t.stack().to_frame(),index=rowvars,columns=colvars,aggfunc='sum')
        if colvars!=[]:
            table.columns = [x[1] for x in table.columns]
        return table
