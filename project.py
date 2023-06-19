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
from .tracker import tracker
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'SimSAD/data')
pd.options.mode.chained_assignment = None

class projection:
    def __init__(self,start_yr = 2020,stop_yr = 2040, base_yr = 2023, chsld_build = False, 
                 ri_build = False, rpa_penetrate = False, chsld_purchase = False,
                 nsa_open_capacity = 0.5, scn_name = 'reference'):
        self.start_yr = start_yr 
        self.stop_yr = stop_yr 
        self.yr = self.start_yr
        self.base_yr = base_yr
        self.chsld_build = chsld_build
        self.ri_build = ri_build
        self.rpa_penetrate = rpa_penetrate
        self.chsld_purchase = chsld_purchase
        self.nsa_open_capacity = nsa_open_capacity
        self.load_params()
        self.init_tracker(scn_name)
        self.milieux = ['none','home','rpa','ri','nsa','chsld']
        self.nmilieux = 6
        return 
    def load_params(self):
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
        self.pop = isq(region=geo)
        self.nregions = self.pop.nregions
        self.last_region = self.nregions + 1
        return 
    def load_grouper(self,set_yr=2016):
        self.grouper = gps(self.pop.nregions,self.pop.count.columns)
        self.grouper.load(set_yr=set_yr)
        return 
    def load_smaf(self,set_yr=2016):
        self.iso = smaf(self.pop.nregions,self.pop.count.columns) 
        self.iso.load(set_yr=set_yr)
        self.nsmaf = self.iso.nsmaf
        self.last_smaf = self.nsmaf + 1
        return 
    def load_chsld(self):
        self.chsld = chsld(opt_build = self.chsld_build, opt_purchase = self.chsld_purchase)
        self.chsld.load_register()
        return
    def load_nsa(self):
        self.nsa = nsa(open_capacity = self.nsa_open_capacity)
        self.nsa.load_register()
    def load_ri(self):
        self.ri = ri(opt_build = self.ri_build)
        self.ri.load_register()
        return 
    def load_rpa(self):
        self.rpa = rpa(opt_penetrate = self.rpa_penetrate)
        self.rpa.load_register()
        return 
    def load_home(self):
        self.home = home()
        self.home.load_register()
        return 
    def load_suppliers(self):
        self.eesad = eesad()
        self.clsc = clsc()
        self.prive = prive()
        self.suppliers = ['eesad','clsc','prive']
        return
    def load_prefs(self):
        self.prefs = prefs()
        return
    def load_financing(self):
        self.msss = msss()
        self.pefsad = pefsad()
        self.ces = ces()
        self.cmd = cmd()
        self.financing = ['pefsad','ces','cmd','msss']
        return
    def set_macro(self, gr_lfp_inf = 0.0, gr_lfp_avq = 0.0, gr_lfp_avd = 0.0, real_int_rate = 0.03, gr_wages = 0.01):
        self.gr_lfp_inf = gr_lfp_inf
        self.gr_lfp_inf = gr_lfp_inf
        self.gr_lfp_inf = gr_lfp_inf
        self.gr_lfp_inf = gr_lfp_inf
        return

    def init_tracker(self, scn_name):
        self.tracker = tracker(scn_name = scn_name)
        show_yr = 2023
        self.tracker.add_entry('pop_region_age','pop','count',['region_id'],
                               ['age'],'sum',show_yr,self.stop_yr)
        self.tracker.add_entry('smaf_region_age','iso','count_smaf',['region_id'],
                               ['smaf'],'sum',show_yr,self.stop_yr)
        self.tracker.add_entry('chsld_users', 'chsld', 'registry',
                            rowvars=['region_id'],
                            colvars=['nb_usagers_tot', 'tx_serv_inf', 'tx_serv_avq'], aggfunc='sum', start_yr=
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
        self.tracker.add_entry('needs_hrs_inf_home', 'home', 'users',
                           rowvars=['region_id'],
                           colvars=['needs_inf'], aggfunc='sum',
                           start_yr=
                           show_yr,
                           stop_yr=self.stop_yr)
        self.tracker.add_entry('needs_hrs_avq_home', 'home', 'users',
                           rowvars=['region_id'],
                           colvars=['needs_avq'], aggfunc='sum',
                           start_yr=
                           show_yr,
                           stop_yr=self.stop_yr)
        self.tracker.add_entry('needs_hrs_avd_home', 'home', 'users',
                           rowvars=['region_id'],
                           colvars=['needs_avd'], aggfunc='sum',
                           start_yr=
                           show_yr,
                           stop_yr=self.stop_yr)
        self.tracker.add_entry('needs_hrs_inf_rpa', 'rpa', 'users',
                           rowvars=['region_id'],
                           colvars=['needs_inf'], aggfunc='sum',
                           start_yr=
                           show_yr,
                           stop_yr=self.stop_yr)
        self.tracker.add_entry('needs_hrs_avq_rpa', 'rpa', 'users',
                           rowvars=['region_id'],
                           colvars=['needs_avq'], aggfunc='sum',
                           start_yr=
                           show_yr,
                           stop_yr=self.stop_yr)
        self.tracker.add_entry('needs_hrs_avd_rpa', 'rpa', 'users',
                           rowvars=['region_id'],
                           colvars=['needs_avd'], aggfunc='sum',
                           start_yr=
                           show_yr,
                           stop_yr=self.stop_yr)
        self.tracker.add_entry('eesad_avd', 'eesad', 'registry',
                            rowvars=['region_id'],
                            colvars=['supply_avd', 'needs_avd'], aggfunc='sum',
                            start_yr=show_yr, stop_yr=self.stop_yr)
        self.tracker.add_entry('prive_avq', 'prive', 'registry',
                            rowvars=['region_id'],
                            colvars=['supply_avq', 'needs_avq'], aggfunc='sum',
                            start_yr=show_yr, stop_yr=self.stop_yr)
        self.tracker.add_entry('ces_users', 'home', 'users',
                            rowvars=['region_id'],
                            colvars=['ces_any','ces_hrs_avd','ces_hrs_avq'],
                            aggfunc='sum',
                            start_yr=show_yr, stop_yr=self.stop_yr)
        self.tracker.add_entry('pefsad_users', 'home', 'users',
                            rowvars=['region_id'],
                            colvars=['pefsad_avd_any','pefsad_avd_hrs'], aggfunc='sum',
                            start_yr=show_yr, stop_yr=self.stop_yr)
        self.tracker.add_entry('total_cost', 'msss', 'registry',
                            rowvars=['region_id'],
                            colvars=['clsc','chsld','ri','nsa','ces','pefsad','cmd',
                    'cah_chsld','cah_ri','cah_nsa','pefsad_usager','total',
                    'gouv','usagers'], aggfunc='sum',
                            start_yr=show_yr, stop_yr=self.stop_yr)
        return
    def run(self):
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

    def init_dispatch(self, init_smafs):
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
        init_smafs['70-91'] = init_smafs[[x for x in range(70,91)]].sum(axis=1)
        init_smafs = init_smafs.loc[:,['18-64','65-69','70-91']]
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
            agent.setup_params(init_prob = iprob, trans_prob = tprob, surv_prob = sprob, wait_count = wait_count,
                               wait_prob_chsld = wprob_chsld, wait_prob_ri = wprob_ri)
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
            self.last_wait = np.zeros((self.nregions, self.nsmaf, nages, self.nmilieux, self.nmilieux))
            self.last_wait[r-1,:,:,:,:] = agent.last_wait
            # full-time equivalent in each living arrangement
            for s in range(1,self.last_smaf):
                for a in gr_ages:
                    self.count.loc[(r,s,a),:] = agent.roster.loc[(s,a),:].values/12
                    self.count_waiting.loc[(r,s,a),:] = agent.waiting_list.loc[(s,a),:].values/12
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
        self.chsld.create_users(self.count['chsld'])
        self.nsa.create_users(self.count['nsa'])
        self.ri.create_users(self.count['ri'])
        self.rpa.create_users(self.count['rpa'])
        self.home.create_users(self.count['none'], self.count['home'])
        return
    def update_users(self):
        self.chsld.update_users()
        self.nsa.update_users()
        self.ri.update_users()
        self.rpa.update_users()
        self.home.update_users()
        return
    def welfare(self):
        self.chsld.users = self.prefs.compute_utility(self.chsld.users)
        #print('chsld = ',self.chsld.users.describe().transpose())
        self.nsa.users = self.prefs.compute_utility(self.nsa.users)
        #print('nsa = ',self.nsa.users.describe().transpose())
        self.ri.users = self.prefs.compute_utility(self.ri.users)
        self.rpa.users = self.prefs.compute_utility(self.rpa.users)
        self.home.users = self.prefs.compute_utility(self.home.users)
        #print('home = ',self.home.users.describe().transpose()[['mean','min',
        #'max']])
        return 

    def chsld_services(self):
        self.chsld.compute_serv_rate()
        self.chsld.compute_costs()
        return
    def nsa_services(self):
        self.nsa.compute_costs()
        return
    def ri_services(self):
        self.ri.compute_serv_rate()
        self.ri.compute_costs()
        return
    def clsc_services(self):
        # determine services
        self.home.users = self.clsc.assign(self.home.users,'home')
        self.rpa.users = self.clsc.assign(self.rpa.users,'rpa')
        self.ri.users = self.clsc.assign(self.ri.users,'ri')
        # adjust services to supply
        self.clsc.compute_supply()
        #self.home.users = self.clsc.cap(self.home.users,'home')
        #self.rpa.users = self.clsc.cap(self.rpa.users,'rpa')
        #self.ri.users = self.clsc.cap(self.ri.users,'ri')
        # determine needs and compute service rate
        self.clsc.compute_needs()
        self.clsc.compute_serv_rate()
        # determine costs
        self.clsc.compute_costs()
        return
    def eesad_services(self):
        self.home.users = self.pefsad.assign(self.home.users,'home')
        self.rpa.users = self.pefsad.assign(self.rpa.users,'rpa')
        self.home.users, self.rpa.users = self.eesad.assign(self.home.users,
                                                          self.rpa.users)
        self.eesad.compute_supply()
        #self.home.users = self.eesad.cap(self.home.users)
        #self.rpa.users = self.eesad.cap(self.rpa.users)
        self.eesad.compute_needs()
        self.eesad.compute_serv_rate()
        self.eesad.compute_costs()
        return
    def private_services(self):
        self.home.users = self.ces.assign(self.home.users)
        if self.yr==2021:
           self.ces.calibrate(self.home.users,
                              self.prive.registry)
        self.prive.assign(self.home.users)
        self.prive.compute_supply()
        #self.home.users = self.prive.cap(self.home.users)
        self.prive.compute_needs()
        self.prive.compute_serv_rate()
        self.prive.compute_costs()
        return
    def cmd_payout(self):
        self.home.users = self.cmd.assign(self.home.users,'home')
        self.rpa.users = self.cmd.assign(self.rpa.users,'rpa')
        self.home.users,self.rpa.users = self.cmd.calibrate(self.home.users,self.rpa.users, self.yr)
        self.cmd.compute_costs(self.home.users, self.rpa.users)
        return
    def finance(self):
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
        print(self.msss.registry.sum(axis=0))
        return
    def compute(self):
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
        self.clsc_services()
        # determine PEFSAD (and ESSAD care)
        self.eesad_services()
        # determine CES (private services)
        self.private_services()
        # determine CMD
        self.cmd_payout()
        # update users with service rate and net oop cost
        self.update_users()
        # compute utility
        self.welfare()
        # compute aggregate costs
        self.finance()
        return  
    
    def next(self):
        self.yr +=1 
        # build new places for institutional settings
        if self.yr>self.base_yr:
            self.chsld.build()
            self.chsld.purchase()
            self.ri.build()
            self.rpa.build()

        return
    def save(self, output_dir):
        self.tracker.save(output_dir)
        return