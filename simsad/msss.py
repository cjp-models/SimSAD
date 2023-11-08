import pandas as pd
import numpy as np
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'simsad/data')


class msss:
    """
    Ministère de la Santé et des Services sociaux

    Cette classe permet de comptabiliser les coûts pour le ministère de la Santé et des Services sociaux.
    """
    def __init__(self):
        self.nregions = 18
        self.last_region = self.nregions + 1
        items = ['clsc','chsld','ri','nsa','ces','pefsad','cmd',
                    'cah_chsld','cah_ri','cah_nsa','pefsad_usager','total',
                    'gouv','usagers']
        self.items = items
        self.registry = pd.DataFrame(index=np.arange(1,self.last_region),columns
        = items)
        self.registry.loc[:, :] = 0.0
        return
    def reset(self):
        self.registry.loc[:,:] = 0.0
        return
    def assign(self, cost, item):
        """
        Cette fonction assigne les coûts des différents items au registre du MSSS.

        Parameters
        ----------
        cost: float
            valeur du coût de l'item
        item: string
            nom de l'item
        """
        self.registry.loc[:,item] = cost * 1e-6
        self.registry.loc[self.registry[item].isna(),item] = 0.0
        return
    def collect(self):
        """
        Cette fonction calcule les coûts agrégés pour le gouvernement, les usagers, ainsi que les coûts totaux.
        """
        self.registry['total'] = self.registry[['clsc','chsld','ri','nsa',
                                                'ces','pefsad','cmd']].sum(
            axis=1)
        self.registry['usagers'] = self.registry[['cah_chsld','cah_ri',
                                                  'cah_nsa',
                                                  'pefsad_usager']].sum(axis=1)
        self.registry['gouv'] = self.registry['total'] - self.registry[
            'usagers']
        return
    def collapse(self, domain = 'registry', rowvars=['region_id'],colvars=[]):
        colvars = self.items
        if domain == 'registry':
                table = self.registry.loc[:,colvars]
        return table
