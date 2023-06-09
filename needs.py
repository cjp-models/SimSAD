import pandas as pd
import numpy as np
import os
from itertools import product
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),'SimSAD/data')
pd.options.mode.chained_assignment = None

class needs:
    def __init__(self):
        reg = pd.read_csv(os.path.join(data_dir,'needs.csv'),
            delimiter=';',low_memory=False,dtype='float64')
        self.inf = reg['inf'].to_list()
        self.avq = reg['avq'].to_list()
        self.avd = reg['avd'].to_list()
        return
