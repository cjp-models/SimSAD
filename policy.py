
class policy:
    def __init__(self):
        self.chsld_build = True
        self.chsld_build_rate = 0.2
        self.ri_build = False
        self.ri_build_rate = 0.2
        self.rpa_penetrate = False
        self.rpa_penetrate_rate = 0.25
        self.rpa_adapt_rate = 0.5
        self.chsld_purchase = True
        self.chsld_purchase_rate = 0.25
        self.nsa_open_capacity = 0.1
        self.chsld_mda = True
        self.infl_construction = 0.01
        self.clsc_cap = True
        self.prive_cap = True
        self.eesad_cap = True
        self.purchase_prive = True
        self.purchase_eesad = True
        self.clsc_inf_rate = 0.25
        self.clsc_avq_rate = 0.25
        self.clsc_avd_rate = 0.25
        self.eesad_avd_rate = 0.25
        self.prive_avq_rate = 0.25
        self.prive_avd_rate = 0.25
        self.chsld_inf_rate = 1.0
        self.chsld_avq_rate = 1.0
        self.ri_avq_rate = 1.0
        self.ri_avd_rate = 1.0
        return