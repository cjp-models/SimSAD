import pandas as pd 
import numpy as np 

# par smaf, pour AVD, AVQ: maison
# heures pour AVD, PEFSAD (par SMAF)
# nombre usagers (par SMAF)
# heures AVQ, iCLSC (par SMAF)
# personnel AVD et AVQ (ETC, enquête)
class eesad: 
    def __init__(self):
        return 
    def load_registry(self):
        # there is a registry by region, there are users for AVD and AVQ by SMAF
        self.registry = pd.DataFrame(index=np.arange(1,19), columns = ['nb_usagers_avd','nb_usagers_avq'])
        self.registry['nb_usagers_avd'] = 100.0
        self.registry['nb_usagers_avq'] = 100.0
        self.registry[['avd_smaf'+str(s) for s in range(1,15)]] = 0.0
        self.registry['avd_smaf1'] = self.registry.loc[:,'nb_usagers_avd']/3
        self.registry['avd_smaf2'] = self.registry.loc[:,'nb_usagers_avd']/3
        self.registry['avd_smaf3'] = self.registry.loc[:,'nb_usagers_avd']/3
        self.registry[['avq_smaf'+str(s) for s in range(1,15)]] = 0.0
        self.registry['avq_smaf2'] = self.registry.loc[:,'nb_usagers_avq']/4
        self.registry['avq_smaf3'] = self.registry.loc[:,'nb_usagers_avq']/4
        self.registry['avq_smaf4'] = self.registry.loc[:,'nb_usagers_avq']/4
        self.registry['avq_smaf5'] = self.registry.loc[:,'nb_usagers_avq']/4
        self.registry['tx_serv_avd'] = 0.0
        self.registry['tx_serv_avq'] = 0.0
        self.registry['nb_etc_avd'] = 1.0
        self.registry['nb_etc_avq'] = 1.0
        self.registry['heures_tot_trav_avd'] = 1500.0
        self.registry['heures_tot_trav_avq'] = 1500.0
        self.registry['share_admin_avd'] = 0.1
        self.registry['share_admin_avq'] = 0.1
        self.needs_avd = [0.61,1.54,1.89,1.75,2,2.04,
                            2.04,2.07,2.07,2.07,2.11,2.07,2.11,2.07]
        self.needs_avq = [0.26,0.27,0.48,0.57,0.67,0.68,
                            1.08,1.24,2.29,2.29,2.61,2.54,2.62,3.08]
        self.days_per_week = 7
        self.days_per_year = 365
        self.weeks_vacation_avd = 4
        self.weeks_vacation_avq = 4
        self.work_days_per_week = 5
        self.time_per_pause = 0.5
        self.nsmafs = 14
        self.nmonths = 12
        self.nregions = 17
        self.weeks_per_year = (self.days_per_year/self.days_per_week)
        self.registry['hours_per_etc_avd'] = self.registry['heures_tot_trav_avd']/self.registry['nb_etc_avd']
        self.registry['hours_per_etc_avq'] = self.registry['heures_tot_trav_avq']/self.registry['nb_etc_avq']
        # to compute take_up rates in first year
        self.take_up_rates_avd = pd.DataFrame(index=np.arange(1,self.nregions+1),columns = np.arange(1,self.nsmafs+1),dtype='float64')
        self.take_up_rates_avq = pd.DataFrame(index=np.arange(1,self.nregions+1),columns = np.arange(1,self.nsmafs+1),dtype='float64')
        return 
    def compute_penetration_rates(self, region_id, nb_eligible_smafs):
        # nb_eligible_smafs is number living at home by SMAF with needs
        for s in range(1,self.nsmafs+1):
            if nb_eligible_smafs[s-1]>0:
                self.take_up_rates_avd.loc[region_id,s] =  self.registry.loc[region_id,'avd_smaf'+str(s)]/nb_eligible_smafs[s-1]
                self.take_up_rates_avq.loc[region_id,s] =  self.registry.loc[region_id,'avq_smaf'+str(s)]/nb_eligible_smafs[s-1]
            else :
                self.take_up_rates_avd.loc[region_id,s] = 0
                self.take_up_rates_avq.loc[region_id,s] = 0   
        return 
    def assign(self, region_id, nb_eligible_smafs):
        for s in range(1,self.nsmafs+1):
            self.registry.loc[region_id,'avd_smaf'+str(s)] = self.take_up_rates_avd.loc[region_id,s] * nb_eligible_smafs[s-1]
            self.registry.loc[region_id,'avq_smaf'+str(s)] = self.take_up_rates_avq.loc[region_id,s] * nb_eligible_smafs[s-1]
        self.registry.loc[region_id,'nb_usagers_avd'] = self.registry.loc[region_id,['avd_smaf'+str(s) for s in range(1,self.nsmafs+1)]].sum()
        self.registry.loc[region_id,'nb_usagers_avq'] = self.registry.loc[region_id,['avq_smaf'+str(s) for s in range(1,self.nsmafs+1)]].sum()
        return 
    def compute_needs(self):
       # effective time per patient avd
        time_table = pd.DataFrame(index=self.registry.index,columns = np.arange(1,15),dtype='float64')
        for s in range(1,15):
            time_table[s] = self.registry['avd_smaf'+str(s)]*self.needs_avd[s-1]
        time_per_usager_avd = time_table.sum(axis=1)
        time_per_usager_avd *= self.days_per_year
        # effective time per patient avq
        for s in range(1,15):
            time_table[s] = self.registry['avq_smaf'+str(s)]*self.needs_avq[s-1]
        time_per_usager_avq = time_table.sum(axis=1)
        time_per_usager_avq *= self.days_per_year
        result = pd.concat([time_per_usager_avd,time_per_usager_avq],axis=1)
        result.columns = ['avd','avq']
        return result
    def compute_supply(self):
        # inf
        time_avd = self.registry['hours_per_etc_avd'].copy()
        # take out indirect care
        time_avd *= (1.0 - self.registry['share_admin_avd'])
        # take out pauses
        time_avd -= self.time_per_pause*(self.weeks_per_year-self.weeks_vacation_avd)*self.work_days_per_week
        # blow up using number of nurses
        time_avd  = time_avd * self.registry['nb_etc_avd']
        ## avq
        time_avq = self.registry['hours_per_etc_avq'].copy()
        # take out pauses
        time_avq -= self.time_per_pause*(self.weeks_per_year-self.weeks_vacation_avq)*self.work_days_per_week
        # take out indirect care
        time_avq *= (1.0 - self.registry['share_admin_avq'])
        # blow up using number of nurses
        time_avq  = time_avq * self.registry['nb_etc_avq']
        result = pd.concat([time_avd,time_avq],axis=1)
        result.columns = ['avd','avq']
        return result
    def compute_serv_rate(self):
        needs = self.compute_needs()
        supply = self.compute_supply()
        self.registry['heures_tot_trav_avd'] = supply['avd']
        self.registry['heures_tot_trav_avq'] = supply['avq']
        self.registry['tx_serv_avd'] = np.where(needs['avd']>0,100.0*(supply['avd']/needs['avd']),np.nan)
        self.registry.loc[self.registry.tx_serv_avd>100.0,'tx_serv_avd'] = 100.0
        self.registry['tx_serv_avq'] = np.where(needs['avq']>0,100.0*(supply['avq']/needs['avq']),np.nan)
        self.registry.loc[self.registry.tx_serv_avq>100.0,'tx_serv_avq']= 100.0
        return


# par SMAF, pour AVQ et infirmiers, maison + RPA + RI (infirmiers)    
# nb usagers, par milieu (Avq et infirmiers)
# heures de services, par milieu (avq et infirmiers)
# personnel: nombre etc, milieu de vie (et par type de soins)  
class clsc:
    def __init__(self):
        self.care_types = ['inf','avq']
        return 
    def load_registry(self):
        # there is a registry by region, there are users for INF, AVD and AVQ by SMAF
        self.registry = pd.DataFrame(index=np.arange(1,19), columns = ['nb_usagers_'+c for c in self.care_types])
        self.registry[['nb_usagers_'+c for c in self.care_types]] = 100.0
        for c in self.care_types:
            self.registry[[c+'_smaf'+str(s) for s in range(1,15)]] = 0.0
        for c in self.care_types:
            smafs = [1,2,3,4,5,6,7]
            for s in smafs:
                self.registry[c+'_smaf'+str(s)] =  self.registry.loc[:,'nb_usagers_'+c]/len(smafs)
            self.registry['tx_serv_'+c] = 0.0
            self.registry['tx_serv_'+c] = 0.0
            self.registry['nb_etc_'+c] = 1.0
            self.registry['heures_tot_trav_'+c] = 1500.0
            self.registry['share_admin_'+c] = 0.1
        self.needs_inf = [0.01,0.02,0.23,0.15,0.29,0.31,0.33,
                            0.38,0.43,0.48,0.58,0.47,0.69,0.95]
        self.needs_avq = [0.26,0.27,0.48,0.57,0.67,0.68,
                            1.08,1.24,2.29,2.29,2.61,2.54,2.62,3.08]
        self.indirect_per_day = 0.4
        self.days_per_week = 7
        self.days_per_year = 365
        self.weeks_vacation_inf = 4
        self.weeks_vacation_avq = 4
        self.work_days_per_week = 5
        self.time_per_pause = 0.5
        self.nsmafs = 14
        self.nmonths = 12
        self.nregions = 17
        self.weeks_per_year = (self.days_per_year/self.days_per_week)
        for c in self.care_types:
            self.registry['hours_per_etc_'+c] = self.registry['heures_tot_trav_'+c]/self.registry['nb_etc_'+c]
        # to compute take_up rates in first year
        self.take_up_rates_inf = pd.DataFrame(index=np.arange(1,self.nregions+1),columns = np.arange(1,self.nsmafs+1),dtype='float64')
        self.take_up_rates_avq = pd.DataFrame(index=np.arange(1,self.nregions+1),columns = np.arange(1,self.nsmafs+1),dtype='float64')
        return 
    def compute_penetration_rates(self, region_id, nb_eligible_smafs_inf, nb_eligible_smafs_avq):
        # nb_eligible_smafs is number living at home by SMAF with needs
        for s in range(1,self.nsmafs+1):
            if nb_eligible_smafs_inf[s-1]>0:
                self.take_up_rates_inf.loc[region_id,s] =  self.registry.loc[region_id,'inf_smaf'+str(s)]/nb_eligible_smafs_inf[s-1]
            else :
                self.take_up_rates_inf.loc[region_id,s] = 0
            if nb_eligible_smafs_avq[s-1]>0:
                self.take_up_rates_avq.loc[region_id,s] =  self.registry.loc[region_id,'avq_smaf'+str(s)]/nb_eligible_smafs_avq[s-1]
            else :
                self.take_up_rates_avq.loc[region_id,s] = 0
        return 
    def assign(self, region_id, nb_eligible_smafs_inf, nb_eligible_smafs_avq):
        for s in range(1,self.nsmafs+1):
            self.registry.loc[region_id,'inf_smaf'+str(s)] = self.take_up_rates_inf.loc[region_id,s] * nb_eligible_smafs_inf[s-1]
            self.registry.loc[region_id,'avq_smaf'+str(s)] = self.take_up_rates_avq.loc[region_id,s] * nb_eligible_smafs_avq[s-1]
        self.registry.loc[region_id,'nb_usagers_inf'] = self.registry.loc[region_id,['inf_smaf'+str(s) for s in range(1,self.nsmafs+1)]].sum()
        self.registry.loc[region_id,'nb_usagers_avq'] = self.registry.loc[region_id,['avq_smaf'+str(s) for s in range(1,self.nsmafs+1)]].sum()
        return 
    def compute_needs(self):
       # effective time per patient avd
        time_table = pd.DataFrame(index=self.registry.index,columns = np.arange(1,15),dtype='float64')
        for s in range(1,15):
            time_table[s] = self.registry['inf_smaf'+str(s)]*self.needs_inf[s-1]
        time_per_usager_inf = time_table.sum(axis=1)
        time_per_usager_inf *= self.days_per_year
        # effective time per patient avq
        for s in range(1,15):
            time_table[s] = self.registry['avq_smaf'+str(s)]*self.needs_avq[s-1]
        time_per_usager_avq = time_table.sum(axis=1)
        time_per_usager_avq *= self.days_per_year
        result = pd.concat([time_per_usager_inf,time_per_usager_avq],axis=1)
        result.columns = ['inf','avq']
        return result
    def compute_supply(self):
        # inf
        time_inf = self.registry['hours_per_etc_inf'].copy()
        # take out indirect care
        time_inf *= (1.0 - self.registry['share_admin_inf'])
        # take out pauses
        time_inf -= self.time_per_pause*(self.weeks_per_year-self.weeks_vacation_inf)*self.work_days_per_week
        # blow up using number of nurses
        time_inf  = time_inf * self.registry['nb_etc_inf']
        ## avq
        time_avq = self.registry['hours_per_etc_avq'].copy()
        # take out indirect care
        time_avq *= (1.0 - self.registry['share_admin_avq'])
        # take out pauses
        time_avq -= self.time_per_pause*(self.weeks_per_year-self.weeks_vacation_avq)*self.work_days_per_week
        # blow up using number of nurses
        time_avq  = time_avq * self.registry['nb_etc_avq']
        result = pd.concat([time_inf,time_avq],axis=1)
        result.columns = ['inf','avq']
        return result
    def compute_serv_rate(self):
        needs = self.compute_needs()
        supply = self.compute_supply()
        for c in self.care_types:
            self.registry['heures_tot_trav_'+c] = supply[c]
            self.registry['tx_serv_'+c] = np.where(needs[c]>0,100.0*(supply[c]/needs[c]),np.nan)
            self.registry.loc[self.registry['tx_serv_'+c]>100.0,'tx_serv_'+c] = 100.0
        return

# autres services achetés, AVQ + soins infirmiers     
class prive:
    def __init__(self):
        return 
    



