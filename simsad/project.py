import pandas as pd
import numpy as np
import os
from itertools import product
from .demo import isq, gps, smaf
from .dispatch import dispatcher
from .chsld import chsld
from .ri import ri
from .rpa import rpa
from .home import home
from .nsa import nsa
from .prefs import prefs
from .eesad import eesad
from .clsc import clsc
from .prive import prive
from .cmd import cmd
from .msss import msss
from .pefsad import pefsad
from .ces import ces
from .policy import policy
from .tracker import tracker
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'simsad/data')
pd.options.mode.chained_assignment = None
import pickle

class projection:
    """
    Modèle de simulation SimSAD.

    Cette classe permet de créer une instance de modèle pour la simulation.

    Parameters
    ----------
    start_yr: int
        année de départ de la simulation (défaut=2020)
    stop_yr: int
        année de fin de la simulation (défaut=2040)
    base_yr: int
        année de départ pour la comptabilisation des résultats (défaut=2023)
    scn_name: string
        nom du scénario (défaut='reference')
    opt_welfare: boolean
        paramètre d'activation du calcul de l'utilité des individus (défaut=False)
    seed: float 
        spécifie la valeur de départ de la séquence aléatoire (défaut=1234)   
    """
    def __init__(self,start_yr = 2020,stop_yr = 2040, base_yr = 2023,
                 scn_policy = None, scn_name = 'reference', opt_welfare =
                 False, seed = 1234):
        self.start_yr = start_yr 
        self.stop_yr = stop_yr 
        self.yr = self.start_yr
        self.base_yr = base_yr
        if scn_policy==None:
            self.policy = policy()
        else :
            self.policy = scn_policy
        self.load_params()
        self.scn_name = scn_name
        self.init_tracker(scn_name)
        self.milieux = ['none','home','rpa','ri','nsa','chsld']
        self.nmilieux = 6
        self.opt_welfare = opt_welfare
        np.random.seed(seed=seed)
        return 
    def load_params(self):
        """
        Chargement des paramètres des différentes composantes du modèle.
        """
        self.load_pop()
        self.load_grouper()
        self.load_smaf()
        self.load_home()
        self.load_rpa()
        self.load_ri()
        self.load_nsa()
        self.load_chsld()
        self.load_suppliers()
        self.load_financing()
        self.load_prefs()
        return 
    def load_pop(self,geo='RSS'):
        """
        Chargement des paramètres démographiques.

        Parameters
        ----------
        geo: str
            unité géographique des projections démographiques (défaut='Québec')
        """
        self.pop = isq(region=geo)
        self.nregions = self.pop.nregions
        self.last_region = self.nregions + 1
        return 
    def load_grouper(self,set_yr=2016):
        """
        Chargement des paramètres d'attribution des groupes de profil de santé (GPS).

        Parameters
        ----------
        set_yr: int
            année de référence (défaut=2016)
        """
        self.grouper = gps(self.pop.nregions,self.pop.count.columns)
        self.grouper.load(set_yr=set_yr)
        return 
    def load_smaf(self,set_yr=2016):
        """
        Chargement des paramètres d'attribution des profil Iso-SMAF.

        Parameters
        ----------
        set_yr: int
            année de référence (défaut=2016)
        """
        self.iso = smaf(self.pop.nregions,self.pop.count.columns) 
        self.iso.load(set_yr=set_yr)
        self.nsmaf = self.iso.nsmaf
        self.last_smaf = self.nsmaf + 1
        return 
    def load_chsld(self):
        self.chsld = chsld(self.policy)
        self.chsld.load_register()
        return
    def load_nsa(self):
        self.nsa = nsa(self.policy)
        self.nsa.load_register()
    def load_ri(self):
        self.ri = ri(self.policy)
        self.ri.load_register()
        return 
    def load_rpa(self):
        self.rpa = rpa(self.policy)
        self.rpa.load_register()
        return 
    def load_home(self):
        self.home = home(self.policy)
        self.home.load_register()
        return 
    def load_suppliers(self):
        self.eesad = eesad(self.policy)
        self.clsc = clsc(self.policy)
        self.prive = prive(self.policy)
        return
    def load_prefs(self):
        self.prefs = prefs()
        return
    def load_financing(self):
        self.msss = msss()
        self.pefsad = pefsad()
        self.ces = ces()
        self.cmd = cmd()
        return
    def init_tracker(self, scn_name):
        """
        Fonction qui spécifie les tableaux de sortie de base dans le modèle.

        Parameters
        ----------
        scn_name: str
            nom du scénario (défaut='reference')
        """
        self.tracker = tracker(scn_name = scn_name)
        show_yr = 2023
        self.tracker.add_entry('pop_region_age','pop','count',['region_id'],
                               ['age'],'sum',show_yr,self.stop_yr)
        self.tracker.add_entry('smaf_region_age','iso','count_smaf',['region_id'],
                               ['smaf'],'sum',show_yr,self.stop_yr)
        self.tracker.add_entry('chsld_users', 'chsld', 'registry',
                            rowvars=['region_id'],
                            colvars=['nb_usagers_tot',
                                     'tx_serv_inf', 'tx_serv_avq','attente_usagers'],
                               aggfunc='sum', start_yr=
                            show_yr,
                            stop_yr=self.stop_yr)
        self.tracker.add_entry('nsa_users', 'nsa', 'registry',
                            rowvars=['region_id'],
                            colvars=['nb_usagers'], aggfunc='sum',
                            start_yr=show_yr,
                            stop_yr=self.stop_yr)
        self.tracker.add_entry('ri_users', 'ri', 'registry', rowvars=['region_id'],
                            colvars=['nb_usagers'], aggfunc='sum',
                            start_yr=show_yr,
                            stop_yr=self.stop_yr)
        self.tracker.add_entry('rpa_users', 'rpa', 'registry',
                            rowvars=['region_id'],
                            colvars=['nb_usagers'], aggfunc='sum',
                            start_yr=show_yr,
                            stop_yr=self.stop_yr)
        self.tracker.add_entry('home_none_users', 'home', 'registry',
                            rowvars=['region_id'],
                            colvars=['nb_usagers_none'], aggfunc='sum',
                            start_yr=show_yr,
                            stop_yr=self.stop_yr)
        self.tracker.add_entry('home_svc_users', 'home', 'registry',
                            rowvars=['region_id'],
                            colvars=['nb_usagers_svc'], aggfunc='sum',
                            start_yr=show_yr,
                            stop_yr=self.stop_yr)
        self.tracker.add_entry('ces_users', 'home', 'users',
                            rowvars=['region_id'],
                            colvars=['ces_any','ces_hrs_avd','ces_hrs_avq'],
                            aggfunc='sum',
                            start_yr=show_yr, stop_yr=self.stop_yr)
        self.tracker.add_entry('pefsad_users', 'home', 'users',
                            rowvars=['region_id'],
                            colvars=['pefsad_avd_any','pefsad_avd_hrs'], aggfunc='sum',
                            start_yr=show_yr, stop_yr=self.stop_yr)
        self.tracker.add_entry('clsc_workforce', 'clsc', 'registry',
                            rowvars=['region_id'],
                            colvars=['nb_etc_inf','nb_etc_avq','nb_etc_avd'],
                               aggfunc='sum',
                            start_yr=show_yr, stop_yr=self.stop_yr)
        self.tracker.add_entry('total_cost', 'msss', 'registry',
                            rowvars=['region_id'],
                            colvars=['clsc','chsld','ri','nsa','ces','pefsad','cmd',
                    'cah_chsld','cah_ri','cah_nsa','pefsad_usager','total',
                    'gouv','usagers'], aggfunc='sum',
                            start_yr=show_yr, stop_yr=self.stop_yr)
        self.tracker.add_entry('clsc_worker_needs', 'clsc', 'registry',
                            rowvars=['region_id'],
                            colvars=['worker_needs_inf','worker_needs_avq','worker_needs_avd'],
                               aggfunc='sum',
                            start_yr=show_yr, stop_yr=self.stop_yr)
        return


    def init_dispatch(self, init_smafs):
        """
        Fonction qui formate les paramètres d'attribution des milieux de vie.

        Parameters
        ----------
        init_smafs: dataframe
            nombre de personnes par profil Iso-SMAF selon la région et le groupe d'âge
        """
        gr_ages = [1,2,3]
        self.init_pars = pd.read_csv(os.path.join(data_dir, 'nb_milieu_vie_init.csv'),
                                     delimiter=';', low_memory=False)
        self.init_pars.set_index(['region_id', 'iso_smaf', 'gr_age'], inplace=True)
        for r in range(1, self.last_region):
            for s in range(1, self.last_smaf):
                for a in gr_ages:
                    self.init_pars.loc[(r, s, a), :] = self.init_pars.loc[(r, s, a), :] / init_smafs.loc[(r, a), s]
        self.pars = pd.read_csv(os.path.join(data_dir, 'transition_mensuelles_milieu_vie.csv'),
                                delimiter=';', low_memory=False)
        self.pars.set_index(['region_id', 'iso_smaf', 'gr_age'], inplace=True)
        self.surv_pars = pd.read_csv(os.path.join(data_dir, 'prob_mensuelles_deces.csv'),
                                     delimiter=';', low_memory=False)
        self.surv_pars.set_index(['region_id', 'iso_smaf', 'gr_age'], inplace=True)
        for c in self.surv_pars.columns:
            self.surv_pars[c] = 1.0 - self.surv_pars[c]
        self.wait_pars_chsld = pd.read_csv(os.path.join(data_dir, 'prob_mv_attente.csv'),
                                           delimiter=';', low_memory=False)
        self.wait_pars_chsld.set_index(['region_id', 'iso_smaf', 'gr_age'], inplace=True)
        self.wait_pars_ri = pd.read_csv(os.path.join(data_dir, 'prob_mv_attente_ri.csv'),
                                        delimiter=';', low_memory=False)
        self.wait_pars_ri.set_index(['region_id', 'iso_smaf', 'gr_age'], inplace=True)
        return

    def dispatch(self):
        """
        Fonction qui attribue les milieux de vie aux personnes.
        """
        # initialize counts
        gr_ages = [1,2,3]
        nages = 3
        tups = list(product(*[np.arange(1,self.last_region),np.arange(1,self.last_smaf),gr_ages]))
        itups = pd.MultiIndex.from_tuples(tups)
        self.count = pd.DataFrame(index=itups,
                                  columns=self.milieux)
        self.count_waiting = pd.DataFrame(index=itups,
                                  columns=self.milieux)
        self.count.columns.names = ['milieu']
        merge_keys = ['region_id','iso_smaf','gr_age']
        self.count.index.names = merge_keys
        self.count_waiting.columns.names = ['milieu']
        self.count_waiting.index.names = merge_keys

        # figure out distribution of smaf by region
        init_smafs = self.iso.collapse(rowvars=['region_id','smaf'],colvars=['age'])
        init_smafs['18-64'] = init_smafs[[x for x in range(18,65)]].sum(axis=1)
        init_smafs['65-69'] = init_smafs[[x for x in range(65,70)]].sum(axis=1)
        init_smafs['70-90'] = init_smafs[[x for x in range(70,91)]].sum(axis=1)
        init_smafs = init_smafs.loc[:,['18-64','65-69','70-90']]
        init_smafs.columns = gr_ages
        init_smafs.columns.names = ['gr_age']
        init_smafs = pd.pivot_table(init_smafs.stack().to_frame(),
                                    index=['region_id','gr_age'],columns=['smaf'],aggfunc=sum)[0]
        # load parameters for transitions
        if self.yr == self.start_yr:
            self.init_dispatch(init_smafs)

            self.last_iprob = np.zeros((self.nregions,self.nsmaf,nages,self.nmilieux))
            self.last_wait = np.zeros((self.nregions,self.nsmaf,nages,self.nmilieux,self.nmilieux))
            for r in range(1,self.last_region):
                for s in range(1,self.last_smaf):
                    for a in gr_ages:
                        self.last_iprob[r-1,s-1,a-1,:] = self.init_pars.loc[(r,s,a),:]
 
        # main algorithm
        for r in range(1,self.last_region):
            agent = dispatcher()
            ismaf = np.zeros((self.nsmaf,nages))
            for s in range(self.nsmaf):
                for a in range(nages):
                    ismaf[s,a] = init_smafs.loc[(r,a+1),s+1]
            agent.init_smaf(ismaf)   
            # load parameters for region 
            iprob = np.zeros((self.nsmaf,nages,self.nmilieux))
            for s in range(self.nsmaf):
                for a in range(nages):
                    iprob[s, a, :] = self.last_iprob[r-1, s, a,:]
            # load parameters for region 
            tprob = np.zeros((self.nsmaf,nages,self.nmilieux**2))
            for s in range(self.nsmaf):
                for a in range(nages):
                    tprob[s,a,:] = self.pars.loc[(r,s+1,a+1),:].values
            sprob = np.zeros((self.nsmaf,nages,self.nmilieux))
            for s in range(self.nsmaf):
                for a in range(nages):
                    sprob[s,a,:] = self.surv_pars.loc[(r,s+1,a+1),:].values
            wprob_chsld = np.zeros((self.nsmaf,nages,self.nmilieux))
            for s in range(self.nsmaf):
                for a in range(nages):
                    wprob_chsld[s,a,:] = self.wait_pars_chsld.loc[(r,s+1,a+1),:].values
            wprob_ri = np.zeros((self.nsmaf,nages,self.nmilieux))
            for s in range(self.nsmaf):
                for a in range(nages):
                    wprob_ri[s,a,:] = self.wait_pars_ri.loc[(r,s+1,a+1),:].values
            # wait
            wait_count = self.last_wait[r-1,:,:,:,:]
            agent.setup_params(init_prob = iprob, trans_prob = tprob,
                               surv_prob = sprob, wait_count = wait_count,
                               wait_prob_chsld = wprob_chsld,
                               wait_prob_ri = wprob_ri)
            # restricting transition to CHSLD for smaf under 10
            if self.yr>(self.base_yr):
                agent.chsld_restriction(self.policy)
            # marginal effect on transition matrix
            if self.yr>=self.base_yr:
                cah_chsld = self.chsld.registry.loc[r,'cah']
                cah_ri = self.ri.registry.loc[r,'cah']
                agent.marginal_effect(self.policy,
                self.prefs.pars, cah_chsld, cah_ri)
            # deal with capacity
            cap_chsld = self.chsld.registry.loc[r,'nb_places_tot']
            cap_ri = self.ri.registry.loc[r, 'nb_places']
            cap_nsa = self.nsa.registry.loc[r,'nb_places']
            cap_rpa = self.rpa.registry.loc[r, 'nb_places_sad']
            agent.setup_capacity(cap_rpa, cap_ri, cap_nsa, cap_chsld)
            # now assign
            agent.assign()
            # collect for output
            agent.collect()
            self.last_iprob[r-1,:,:,:] = agent.last_state
            self.last_wait = np.zeros((self.nregions, self.nsmaf, nages,
                                       self.nmilieux, self.nmilieux))
            self.last_wait[r-1,:,:,:,:] = agent.last_wait
            # full-time equivalent in each living arrangement
            for s in range(1,self.last_smaf):
                for a in gr_ages:
                    self.count.loc[(r,s,a),:] = agent.roster.loc[(s,a),:].values/12
                    self.count_waiting.loc[(r,s,a),:] = \
                        agent.waiting_list.loc[(s,a),:].values/12
            self.count.clip(lower=0.0,inplace=True)
            self.count_waiting.clip(lower=0.0,inplace=True)
            # save matrices of number of cases
            nb_usagers = np.zeros((self.nsmaf,self.nmilieux))
            nb_waiting = self.count_waiting.loc[(r,),:].sum().values
            for s in range(1,self.last_smaf):
                nb_usagers[s-1,:] = self.count.loc[(r,s,),:].sum(axis=0)
            # people waiting in CHSLD (5) go to NSA (4)
            self.chsld.assign(nb_usagers[:,5],nb_waiting[5],r)
            self.nsa.assign(nb_usagers[:,4],r)
            self.ri.assign(nb_usagers[:,3],nb_waiting[3],r)
            self.rpa.assign(nb_usagers[:,2],nb_waiting[2],r)
            self.home.assign(nb_usagers[:,0], nb_usagers[:,1], nb_waiting[1], r)
 
        # create user sets
        self.create_users()
        return

    def create_users(self):
        """
        Fonction qui crée les bassins d'individus pour l'ensemble des milieux de vie.
        """
        self.chsld.create_users(self.count['chsld'])
        self.nsa.create_users(self.count['nsa'])
        self.ri.create_users(self.count['ri'])
        self.rpa.create_users(self.count['rpa'])
        self.home.create_users(self.count['none'], self.count['home'])
        return
    def update_users(self):
        """
        Fonction qui met à jour les caractéristiques des individus dans les bassins pour l'ensemble des milieux de vie.
        """
        self.chsld.update_users()
        self.nsa.update_users()
        self.ri.update_users()
        self.rpa.update_users()
        self.home.update_users()
        return
    def welfare(self):
        """
        Fonction qui calcule l'utilité des individus.
        """
        self.chsld.users = self.prefs.compute_utility(self.chsld.users)
        self.nsa.users = self.prefs.compute_utility(self.nsa.users)
        self.ri.users = self.prefs.compute_utility(self.ri.users)
        self.rpa.users = self.prefs.compute_utility(self.rpa.users)
        self.home.users = self.prefs.compute_utility(self.home.users)
        return

    def chsld_services(self):
        self.chsld.compute_supply()
        self.chsld.compute_costs()
        return
    def nsa_services(self):
        self.nsa.compute_costs()
        return
    def ri_services(self):
        self.ri.compute_supply()
        self.ri.compute_costs()
        return
    def clsc_services_assign(self):
        # determine services
        self.home.users = self.clsc.assign(self.home.users,'home',
                                           self.policy, self.yr)
        self.rpa.users = self.clsc.assign(self.rpa.users,'rpa',self.policy,self.yr)
        self.ri.users = self.clsc.assign(self.ri.users,'ri',self.policy,self.yr)
        return
    def clsc_services_cap(self):
        self.clsc.compute_supply(self.yr)
        if self.policy.clsc_cap:
            self.home.users = self.clsc.cap(self.home.users,'home')
            self.rpa.users = self.clsc.cap(self.rpa.users,'rpa')
            self.ri.users = self.clsc.cap(self.ri.users,'ri')
        # determine costs
        self.clsc.compute_costs()
        return
    def eesad_services_assign(self):
        self.home.users = self.pefsad.assign(self.home.users,'home')
        self.rpa.users = self.pefsad.assign(self.rpa.users,'rpa')
        self.home.users, self.rpa.users = self.eesad.assign(self.home.users,
                                                          self.rpa.users)
        return
    def eesad_services_cap(self):
        avq_sold_clsc = self.clsc.registry.loc[:,['hrs_sa_avq_eesad_home','hrs_sa_avq_eesad_rpa']].sum(axis=1)
        avd_sold_clsc = self.clsc.registry.loc[:,['hrs_sa_avd_eesad_home','hrs_sa_avd_eesad_rpa']].sum(axis=1)
        self.eesad.compute_supply(avq_sold_clsc,avd_sold_clsc)
        if self.policy.eesad_cap:
            self.home.users, self.rpa.users = self.eesad.cap(self.home.users, self.rpa.users)
        self.eesad.compute_costs()
        return

    def private_services_assign(self):
        self.home.users = self.ces.assign(self.home.users)
        if self.yr==2021:
           self.ces.calibrate(self.home.users,
                              self.prive.registry)
        self.prive.assign(self.home.users)
        return
    def private_services_cap(self):
        self.prive.compute_supply()
        if self.policy.prive_cap:
            self.home.users = self.prive.cap(self.home.users)
        self.prive.compute_costs()
        return
    def cmd_payout(self):
        self.home.users = self.cmd.assign(self.home.users,'home')
        self.rpa.users = self.cmd.assign(self.rpa.users,'rpa')
        self.home.users,self.rpa.users = self.cmd.calibrate(self.home.users,self.rpa.users, self.yr)
        self.cmd.compute_costs(self.home.users, self.rpa.users)
        return

    def finance(self):
        """
        Fonction qui collige les coûts pour tous les milieux de vie et les programmes de financement.
        """
        # add total program costs
        items = ['clsc','chsld','ri','nsa','ces','pefsad','cmd',
                    'cah_chsld','cah_ri','cah_nsa','pefsad_usager','total',
                    'gouv','usagers']
        self.msss.assign(self.clsc.registry['cout_total'],'clsc')
        self.msss.assign(self.chsld.registry['cout_total'],'chsld')
        self.msss.assign(self.nsa.registry['cout_total'],'nsa')
        self.msss.assign(self.ri.registry['cout_total'],'ri')
        self.msss.assign(self.prive.registry['cout_total'],'ces')
        self.msss.assign(self.eesad.registry['cout_total'],'pefsad')
        self.msss.assign(self.cmd.registry['cout_total'],'cmd')
        # add user fees
        cah_chsld = 12.0* self.chsld.users.groupby('region_id').apply(lambda d:
                                                                (d['cost']*d['wgt']).sum())
        self.msss.assign(cah_chsld,'cah_chsld')

        cah_nsa = 12.0* self.nsa.users.groupby('region_id').apply(lambda d:
                                                                (d['cost']*d['wgt']).sum())
        self.msss.assign(cah_nsa,'cah_nsa')
        cah_ri = 12.0* self.ri.users.groupby('region_id').apply(lambda d:
                                                                (d['cost']*d['wgt']).sum())
        self.msss.assign(cah_ri,'cah_ri')
        user_pefsad_home = 12.0 * self.home.users.groupby('region_id').apply(
            lambda d: (d['cost']*d['wgt']).sum())
        user_pefsad_rpa = 12.0 * self.rpa.users.groupby('region_id').apply(
            lambda d: (d['cost']*d['wgt']).sum())
        user_pefsad = user_pefsad_home + user_pefsad_rpa
        self.msss.assign(user_pefsad,'pefsad_usager')
        self.msss.collect()
        return

    def run(self):
        """
        Fonction déclenchant le lancement de la simulation.
        """
        togo = self.stop_yr - self.start_yr + 1
        while togo>0:
            print(self.yr)
            self.compute()
            self.tracker.log(self,self.yr)
            togo -=1
            if togo>0:
                self.next()
        self.save('output')
        return
    def compute(self):
        """
        Fonction qui appelle l'ensemble des fonctions permettant de calculer les résultats du modèle.
        """
        # exogeneous needs composition at aggregate level (region, age, smaf)
        self.pop.evaluate(self.yr)
        self.grouper.evaluate(self.pop,yr=self.yr)
        self.iso.evaluate(self.grouper)
        # now assign users to each living arrangement, dynamically within year
        self.dispatch()
        # CHSLD
        self.chsld_services()
        # RI
        self.ri_services()
        # NSA
        self.nsa_services()
        # determine services SAD offered by CLSC
        self.clsc_services_assign()
        # determine PEFSAD (and ESSAD care)
        self.eesad_services_assign()
        # determine CES (private services)
        self.private_services_assign()

        # determine services SAD offered by CLSC
        self.clsc_services_cap()
        # determine PEFSAD (and ESSAD care)
        self.eesad_services_cap()
        # determine CES (private services)
        self.private_services_cap()

        # determine CMD
        self.cmd_payout()
        # update users with service rate and net oop cost
        self.update_users()
        # compute utility
        if self.opt_welfare:
            self.welfare()
        # compute aggregate costs
        self.finance()
        return  
    
    def next(self):
        self.yr +=1 
        
        # workforce adjustments (chsld, ri done with build)
        if self.yr<=self.base_yr:
            self.clsc.workforce(before_base_yr=True)
            self.eesad.workforce(before_base_yr=True)
            self.prive.workforce(before_base_yr=True)
        else:
            self.clsc.workforce()
            self.eesad.workforce()
            self.prive.workforce()

        # build new places for institutional settings
        if self.yr>self.base_yr:
            self.chsld.build()
            self.chsld.purchase()
            self.ri.build()
            self.rpa.build()
        return
    def save(self, output_dir):
        """
        Fonction qui permet de sauvegarder les données d'un scénario.

        Parameters
        ----------
        output_dir: str
            sentier vers le dossier de sauvegarde des données.
        """
        self.tracker.save(output_dir, self.policy)
        with open(os.path.join(output_dir,self.scn_name+'.pkl'), 'wb') as f:
            pickle.dump(self, f)
        return
    def load(self):
        """
        Fonction qui permet de charger les données d'un scénario préalablement sauvegardé.
        """
        with open(os.path.join('output',self.scn_name+'.pkl'), 'rb') as f:
            this = pickle.load(f)
        return this