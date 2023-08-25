import numpy as np 
import pandas as pd 
import os 
from functools import partial
from itertools import product
import multiprocessing as mp 
from scipy.optimize import minimize 
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'SimSAD/data')

class prefs:
    def __init__(self):
        pars = pd.read_csv(os.path.join(data_dir,'params_prefs_smaf.csv'),delimiter=';',low_memory=False,header=None)
        pars_labels = ['rpa','ri','chsld','public','profit','tx_serv_inf','tx_serv_avq','tx_serv_avd','wait_time','cost']
        pars['vars'] = pars_labels
        pars.set_index('vars',inplace=True)
        pars.columns = np.arange(1,15)
        pars.loc['nonprofit',:] = 0.0
        pars.loc['home',:] = 0.0
        self.pars = pars
        return 
    def utility(self,row):
        u = 0.0
        u += self.pars.loc[row['milieu'],row['smaf']]
        u += self.pars.loc[row['supplier'],row['smaf']]
        u += self.pars.loc['tx_serv_inf',row['smaf']]*row['tx_serv_inf']
        u += self.pars.loc['tx_serv_avq',row['smaf']]*row['tx_serv_avq']
        u += self.pars.loc['tx_serv_avd',row['smaf']]*row['tx_serv_avd']
        u += self.pars.loc['wait_time',row['smaf']]*row['wait_time']
        u += self.pars.loc['cost',row['smaf']]*row['cost']*1e-3
        # express in money metric
        #u = u/self.pars.loc['cost',row['smaf']]
        return u
    def compute_utility(self,users):
        users['utility'] = users.apply(self.utility,axis=1)
        
        users['utility_in_dollars'] = 0.0
        for s in range(1,15):
            cond = (users['iso_smaf']==s)
            users.loc[cond,'utility_in_dollars'] = 1000*users.loc[cond,'utility']/ -self.pars.loc['cost',s]

        return users





