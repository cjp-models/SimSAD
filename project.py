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
from .prefs import prefs
from .suppliers import eesad, clsc, prive
from .financing import msss, pefsad, ces, cmd
from .tracker import tracker
from functools import partial
#from alive_progress import alive_bar
import time 
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'SimSAD/data')
pd.options.mode.chained_assignment = None

class projection:
    def __init__(self,start_yr = 2020,stop_yr = 2040, base_yr = 2023, chsld_build = False, 
                 ri_build = False, rpa_build = False, chsld_purchase = False):
        self.start_yr = start_yr 
        self.stop_yr = stop_yr 
        self.yr = self.start_yr
        self.base_yr = base_yr
        self.chsld_build = chsld_build
        self.ri_build = ri_build
        self.rpa_build = rpa_build
        self.chsld_purchase = chsld_purchase
        self.load_params()
        self.init_tracker()
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
    def load_ri(self):
        self.ri = ri(opt_build = self.ri_build)
        self.ri.load_register()
        return 
    def load_rpa(self):
        self.rpa = rpa(opt_build = self.rpa_build)
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
    def init_tracker(self):
        self.tracker = tracker() 
        self.tracker.add_entry('pop_region_age','pop',['region_id'],['age'],'sum',self.start_yr,self.stop_yr)
        return 
    def run(self):
        togo = self.stop_yr - self.start_yr + 1
        while togo>0:
            self.compute()
            self.tracker.log(self,self.yr)
            togo -=1
            if togo>0:
                self.next()
        return
    def dispatch(self):
        # allocate to milieu 
        tups = list(product(*[np.arange(1,self.last_region),np.arange(1,self.last_smaf),[1,2,3]]))
        self.count = pd.DataFrame(index=pd.MultiIndex.from_tuples(tups),
                                  columns=['none','home','rpa','ri','nsa','chsld'])
        self.count_waiting = pd.DataFrame(index=pd.MultiIndex.from_tuples(tups),
                                  columns=['none','home','rpa','ri','nsa','chsld'])

        self.count.columns.names = ['milieu']
        self.count.index.names = ['region_id','iso_smaf','gr_age']
        self.count_waiting.columns.names = ['milieu']
        self.count_waiting.index.names = ['region','smaf','gr_age']
        # figure out distribution of smaf by region
        init_smafs = self.iso.collapse(rowvars=['region_id','smaf'],colvars=['age'])
        init_smafs['18-64'] = init_smafs[[x for x in range(18,65)]].sum(axis=1)
        init_smafs['65-69'] = init_smafs[[x for x in range(65,70)]].sum(axis=1)
        init_smafs['70-91'] = init_smafs[[x for x in range(70,91)]].sum(axis=1)
        init_smafs = init_smafs.loc[:,['18-64','65-69','70-91']]
        init_smafs.columns = [1,2,3]
        init_smafs.columns.names = ['gr_age']
        init_smafs = pd.pivot_table(init_smafs.stack().to_frame(),
                                    index=['region_id','gr_age'],columns=['smaf'],aggfunc=sum)[0]
        gr_ages = [1,2,3]
        nages = 3
        # load parameters for transitions
        if self.yr == self.start_yr:
            self.init_pars = pd.read_csv(os.path.join(data_dir,'nb_milieu_vie_init.csv'),
                delimiter=';',low_memory=False)
            self.init_pars.set_index(['region_id','iso_smaf','gr_age'],inplace=True)
            self.cap_nsa = self.init_pars.groupby('region_id').sum()['nsa']
            for r in range(1,self.last_region):
                for s in range(1,self.last_smaf):
                    for a in gr_ages:
                        self.init_pars.loc[(r,s,a),:] = self.init_pars.loc[(r,s,a),:]/init_smafs.loc[(r,a),s]
            self.pars = pd.read_csv(os.path.join(data_dir,'transition_mensuelles_milieu_vie.csv'),
                delimiter=';',low_memory=False)
            self.pars.set_index(['region_id','iso_smaf','gr_age'],inplace=True)
            self.surv_pars = pd.read_csv(os.path.join(data_dir,'prob_mensuelles_deces.csv'),
                delimiter=';',low_memory=False)
            self.surv_pars.set_index(['region_id','iso_smaf','gr_age'],inplace=True)
            for c in self.surv_pars.columns:
                self.surv_pars[c] = 1.0 - self.surv_pars[c]
            self.wait_pars = pd.read_csv(os.path.join(data_dir,'prob_mv_attente.csv'),
                delimiter=';',low_memory=False)
            self.wait_pars.set_index(['region_id','iso_smaf','gr_age'],inplace=True)
            self.last_iprob = np.zeros((self.nregions,self.nsmaf,nages,self.nmilieux))
            self.last_wait = np.zeros((self.nregions,self.nsmaf,nages,self.nmilieux,self.nmilieux))
            for r in range(1,self.last_region):
                for s in range(1,self.last_smaf):
                    for a in gr_ages:
                        self.last_iprob[r-1,s-1,a-1,:] = self.init_pars.loc[(r,s,a),:]
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
            wprob = np.zeros((self.nsmaf,nages,self.nmilieux))
            for s in range(self.nsmaf):
                for a in range(nages):
                    wprob[s,a,:] = self.wait_pars.loc[(r,s+1,a+1),:].values
            # wait
            wait_count = self.last_wait[r-1,:,:,:,:]
            agent.setup_params(init_prob = iprob, trans_prob = tprob, surv_prob = sprob, wait_count = wait_count,
                               wait_prob = wprob)
            # deal with capacity
            if r>=17:
                cap_chsld  = self.chsld.registry.loc[17,'nb_places_tot']
            else :
                cap_chsld = self.chsld.registry.loc[r,'nb_places_tot']
            if r>=17:
                cap_ri = 0
            else :
                cap_ri = self.ri.registry.loc[r, 'nb_places']
            if r>=17:
                cap_nsa = 0
            else :
                cap_nsa = self.cap_nsa[r]
            if r>=17:
                cap_rpa = 0
            else :
                cap_rpa = self.rpa.registry.loc[r, 'nb_places']
            agent.setup_capacity(cap_rpa, cap_ri, cap_nsa, cap_chsld)
            agent.assign()
            agent.collect()
            self.last_iprob[r-1,:,:,:] = agent.last_state
            self.last_wait = np.zeros((self.nregions, self.nsmaf, nages, self.nmilieux, self.nmilieux))
            self.last_wait[r-1,:,:,:,:] = agent.last_wait
            # full-time equivalent in each living arrangement
            for s in range(1,self.last_smaf):
                for a in [1,2,3]:
                    self.count.loc[(r,s,a),:] = agent.roster.loc[(s,a),:].values/12
                    self.count_waiting.loc[(r,s,a),:] = agent.waiting_list.loc[(s,a),:].values/12
            # save matrices of number of cases
            nb_usagers = np.zeros((self.nsmaf,self.nmilieux))
            nb_waiting = self.count_waiting.loc[(r,),:].sum().values
            for s in range(1,self.last_smaf):
                nb_usagers[s-1,:] = self.count.loc[(r,s,),:].sum(axis=0)
            # people waiting in CHSLD (5) go to NSA (4)
            self.chsld.assign(nb_usagers[:,5],nb_waiting[5],r)
            #self.chsld.create_users(self.count['chsld'])
            # people in other milieux
            self.ri.assign(nb_usagers[:,3],nb_waiting[3],r)
            #self.ri.create_users(self.count['ri'])
            self.rpa.assign(nb_usagers[:,2],nb_waiting[2],r)
            self.rpa.create_users(self.count['rpa'])
            self.home.assign(nb_usagers[:,0], nb_usagers[:,1], nb_waiting[1], r)
            self.home.create_users(self.count['none'],self.count['home'])
        return 
    
    def welfare(self):
        self.chsld.users = self.prefs.compute_utility(self.chsld.users)
        self.ri.users = self.prefs.compute_utility(self.ri.users)
        self.rpa.users = self.prefs.compute_utility(self.rpa.users)
        self.home.users = self.prefs.compute_utility(self.home.users)
        return 


    def compute(self):
        # exogeneous needs composition at aggregate level (region, age, smaf)
        self.pop.evaluate(self.yr)
        self.grouper.evaluate(self.pop,yr=self.yr)
        self.iso.evaluate(self.grouper)

        # now assign users to each living arrangement
        self.dispatch()

        # determine services SAD
        self.home.users = self.clsc.assign(self.home.users,'home')
        self.rpa.users = self.clsc.assign(self.rpa.users,'rpa')

        print(self.home.users.loc[self.home.users.any_svc,['clsc_inf_any','clsc_avq_any','clsc_avd_any']].mean())
        print(self.rpa.users[['clsc_inf_any','clsc_avq_any','clsc_avd_any']].mean())

        # compute aggregate service rates
        self.chsld.compute_serv_rate()
        #self.ri.compute_serv_rate()
        #self.rpa.compute_serv_rate()

        # compute utility
        #self.welfare()
        #print(self.chsld.users['utility'].mean(),self.ri.users[
        # 'utility'].mean(),self.rpa.users['utility'].mean(),self.home.users['utility'].mean())

        # create the roster for computing use, cost and utility
        # determine supply and use of SAD, Home + RPA + RI inf
        # determine services for each users
        #self.create_users()
        #print(len(self.users))

        #self.sad()

        # compute user costs (to users)

        # compute utility for users

        # cleanup 
        
        return  
    
    def next(self):
        self.yr +=1 
        # build new places for institutional settings
        if self.yr>self.base_yr:
            self.chsld.build()
            self.chsld.purchase()
        #self.ri.build()
        #self.rpa.build()
        # labor force transitions for Inf, AVQ
        
        return 