import numpy as np 
import pandas as pd 
import os 
from itertools import product
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'SimSAD/data')
from numba import njit, float64, int64, boolean
from numba.types import Tuple

class dispatcher:
    def __init__(self):
        self.setup_milieux()
        self.setup_ages()
        return
    def setup_milieux(self):
        self.milieux = ['none','home','rpa','ri','nsa','chsld']
        # number of milieux
        self.n = len(self.milieux)
        return
    def setup_ages(self):
        self.gr_ages = [1,2,3]
        self.na = 3
        return
    def setup_capacity(self,n_rpa, n_ri, n_nsa, n_chsld):
        self.n_cap = np.zeros(self.n)
        self.n_cap[0] = 100e3
        self.n_cap[1] = 100e3
        self.n_cap[2] = n_rpa
        self.n_cap[3] = n_ri
        self.n_cap[4] = n_nsa
        self.n_cap[5] = n_chsld
        return
    def setup_params(self, init_prob, trans_prob, surv_prob, wait_count, wait_prob):
        # number of months (includes an initial state)
        self.m = 13
        # number of smafs
        self.ns = 14
        # number of age groups
        self.na = 3
        # initial probabilities
        self.pi0 = init_prob
        # transition probabilities across states (conditional on survival)
        self.pi = np.zeros((self.ns,self.na,self.n,self.n))
        for s in range(self.ns):
            for a in range(self.na):
                self.pi[s,a,:,:] = trans_prob[s,a,:].reshape((self.n,self.n))
        # survival probability (month-to-month)
        self.ss = surv_prob
        self.wait_init = wait_count
        self.wprob = wait_prob
        return
    def init_smaf(self,smafs):
        self.smafs = smafs
        self.nsmafs = np.sum(self.smafs)
        return
    def init_state(self):
        self.count_states = np.zeros((self.ns,self.na,self.n,self.m))
        self.count_wait = np.zeros((self.ns,self.na,self.n,self.n,self.m))
        self.count_wait[:,:,:,:,0] = self.wait_init
        for s in range(self.ns):
            for a in range(self.na):
                    self.count_states[s,a,:,0] = self.smafs[s,a] * self.pi0[s,a,:]
        for n in range(self.n-1,-1,-1):
            nusers = np.sum(self.count_states[:, :, n, 0])
            avail_spots = max(self.n_cap[n] - nusers,0)
            if avail_spots>0:
                for s in range(self.ns-1,-1,-1):
                    for a in range(self.na-1,-1,-1):
                        for j in range(self.n-1,-1,-1):
                            waiters = self.count_wait[s,a,j,n,0]
                            if avail_spots > waiters:
                                self.count_states[s,a,n,0] += waiters
                                self.count_wait[s,a,j,n,0] = 0
                                avail_spots -= waiters
                            else :
                                self.count_states[s,a,n,0] += avail_spots
                                self.count_wait[s,a,j,n,0] -= avail_spots
                                avail_spots = 0

        # once checked waiting list, deal with excess users
        for n in range(self.n-1,0,-1):
            nusers = np.sum(self.count_states[:,:,n,0]) + np.sum(self.count_wait[:,:,n,:,0])
#            print(n, nusers, self.n_cap[n], nusers - self.n_cap[n])
            if nusers > self.n_cap[n]:
                excess = max(nusers - self.n_cap[n],0)
                for s in range(self.ns):
                    for a in range(self.na):
                        users = self.count_states[s, a, n, 0]
                        if excess <= users:
                            self.count_states[s,a,n,0] -= excess
                            #self.count_states[s,a,n-1,0] += excess
                            self.count_wait[s,a,n-1,n,0] += excess
                            excess = 0
                        else :
                            self.count_states[s,a,n,0] -= users
                            #self.count_states[s,a,n-1,0] += users
                            self.count_wait[s,a,n-1,n,0] += users
                            excess -= users
#        print('after constraints')
        for n in range(self.n-1,-1,-1):
            nusers = np.sum(self.count_states[:, :, n, 0]) + np.sum(self.count_wait[:,:,n,:,0])
            avail_spots = max(self.n_cap[n] - nusers,0)
#            print(n, nusers, self.n_cap[n], avail_spots)

        return
    def next_state(self,m):
        self.count_states[:,:,:,m+1], self.count_wait[:,:,:,:,m+1] = transition(self.count_states[:,:,:,m],self.count_wait[:,:,:,:,m], self.pi,
                     self.ss, self.n_cap, self.wprob)
        return
    def assign(self):
        self.init_state()
        for m in range(self.m-1):
            self.next_state(m)
        return
    def collect(self):
        # number of person-month in each milieux by smaf
        roster = np.zeros((self.ns, self.na, self.n))
        waiting_list = np.zeros((self.ns, self.na, self.n))
        for s in range(self.ns):
            for a in range(self.na):
                for n in range(self.n):
                    roster[s,a,n] = np.sum(self.count_states[s,a,n,1:])
                    roster[s,a,n] += np.sum(self.count_wait[s,a,n,:,1:])
                    waiting_list[s,a,n] = np.sum(self.count_wait[s,a,:,n,1:])
        tups = list(product(*[np.arange(1,self.ns+1),[1,2,3]]))
        self.roster = pd.DataFrame(columns=self.milieux,index=pd.MultiIndex.from_tuples(tups))
        for c in self.roster.columns:
            self.roster[c] = 0.0
        self.waiting_list = pd.DataFrame(columns=self.milieux,index=pd.MultiIndex.from_tuples(tups))
        for c in self.waiting_list.columns:
            self.waiting_list[c] = 0.0
        for s in range(self.ns):
            for a in range(self.na):
                self.roster.loc[(s+1,a+1), :] = roster[s,a, :]
                self.waiting_list.loc[(s+1,a+1), :] = waiting_list[s,a,:]
        self.last_state = np.zeros((self.ns,self.na,self.n))
        self.last_wait  = np.zeros((self.ns,self.na,self.n, self.n))
        for s in range(self.ns):
            for a in range(self.na):
                for n in range(self.n):
                    self.last_state[s,a,n] = self.count_states[s,a,n,12]
                    for j in range(self.n):
                        self.last_wait[s,a,j,n] = self.count_wait[s,a,j,n,12]
        for s in range(self.ns):
            for a in range(self.na):
                if np.sum(self.last_state[s,a,:])>0:
                    self.last_state[s,a,:] = self.last_state[s,a,:]/np.sum(self.last_state[s,a,:])
                else:
                    self.last_state[s,a,:] = 0.0
        return

@njit(Tuple((float64[:,:,:],float64[:,:,:,:]))(float64[:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:], float64[:], float64[:,:,:]))
def transition(state, wait, pi, ss, ncaps, wprob):
    ns, na, nn = state.shape
    next_state = np.zeros(state.shape) 
    next_wait = np.zeros(wait.shape)
    # start with CHSLD, go down
    for n in range(nn-1,-1,-1):
        # figure out how many stay (some die)j = n
        for s in range(ns-1,-1,-1):
            for a in range(na-1,-1,-1):
                next_state[s,a,n] = pi[s,a,n,n] * ss[s,a,n] * state[s,a,n]
        # figure out how many want to enter and from where (upon surviving)
        appl = np.zeros((ns,na,nn))
        for s in range(ns-1,-1,-1):
            for a in range(na-1,-1,-1):
                for j in range(nn-1,-1,-1):
                    if j!=n:
                        appl[s,a,j] = pi[s,a,j,n] * ss[s,a,j] * state[s,a,j]
        # next year's waiting list with those already on it, but account for survival
        for s in range(ns-1,-1,-1):
            for a in range(na-1,-1,-1):
                for j in range(nn-1,-1,-1):
                      next_wait[s,a,j,n] = wait[s,a,j,n] * ss[s,a,j]
        # currently those who have a spot are those who stay and those on waiting list in that state
        nstay = np.sum(next_state[:, :, n])
        nstay += np.sum(next_wait[:,:,n,:])
        # figure out whether spots available (should always be positive because of mortality)
        avail_spots = max(ncaps[n] - nstay,0)
        # prioritize those on waiting list for that state, proceed from worse SMAF and oldest age
        pej = np.zeros(nn)
        if avail_spots>0:
            for s in range(ns-1,-1,-1):
                for a in range(na-1,-1,-1):
                    # need adapt to use probabilities, but add to ranks and reduce waiting list
                    if n == nn - 1:
                        pw = np.sum(next_wait[s, a, :, n])
                        if pw>0:
                            pj = next_wait[s, a, :, n]/np.sum(next_wait[s, a, :, n])
                            pe = avail_spots/np.sum(next_wait[s, a, :, n])
                        else :
                            pj = np.zeros(nn)
                            pe = 1.0
                        for j in range(nn):
                            if pj[j]>0:
                                pej[j] = wprob[s, a, j]*pe/pj[j]
                            else :
                                pej[j] = 1.0
                    else:
                        pej[:] = 1.0
                    for j in range(nn-1,-1,-1):
                        # not currently in n
                        if j!=n:
                            # are there more spots than people waiting for n in that state j
                            # if in chsld, use probs
                            w = pej[j]*next_wait[s,a,j,n]
                            if avail_spots>=w:
                                # if so, let them enter, reduce number of spots and empty waiting list
                                next_state[s,a,n] += w
                                avail_spots -= w
                                next_wait[s,a,j,n] -= w
                            else :
                                # if not enough space, only allow up to capacity (could be zero spots)
                                next_state[s,a,n] += avail_spots
                                next_wait[s,a,j,n] -= avail_spots
                                avail_spots = 0
        # now if spots still available, let those who apply enter
        if avail_spots > 0:
            # start again from last SMAF and oldest
            for s in range(ns-1,-1,-1):
                for a in range(na-1,-1,-1):
                    for j in range(nn-1,-1,-1):
                        # if not in n already
                        if j != n:
                            # if enough space to accept everyone, add to state and reduce application pool                                          
                            if avail_spots>=appl[s,a,j]:
                                next_state[s,a,n] += appl[s,a,j]
                                avail_spots -= appl[s,a,j]
                                appl[s,a,j] = 0   
                            # if not enough space, only allow up to capacity (could be zero)
                            else:
                                next_state[s,a,n] += avail_spots
                                appl[s,a,j] -= avail_spots
                                avail_spots = 0
        # those left applying are moving to waiting list, add them 
        for s in range(ns-1,-1,-1):
            for a in range(na-1,-1,-1):
                for j in range(nn-1,-1,-1):
                  next_wait[s,a,j,n] += appl[s,a,j]
    # go to next n, coming down
    return next_state, next_wait
