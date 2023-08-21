
class policy:
    def __init__(self):
        self.chsld_build = True
        self.chsld_build_rate = 0.2
        self.ri_build = True
        self.ri_build_rate = 0.2
        self.rpa_penetrate = False
        self.rpa_penetrate_rate = 0.25
        self.rpa_adapt_rate = 0.5
        self.chsld_purchase = True
        self.chsld_purchase_rate = 0.25
        self.nsa_open_capacity = 0.06
        self.chsld_mda = True
        self.infl_construction = 0.01
        self.interest_rate = 0.03
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
        self.delta_inf_rate = 0.0
        self.delta_avq_rate = 0.0
        self.delta_avd_rate = 0.0
        self.delta_cah_chsld  = 0.0
        self.delta_cah_ri  = 0.0
        self.clsc_shift_avq_eesad = 0.0
        self.clsc_shift_avq_prive = 0.0
        self.clsc_shift_avd_eesad = 0.0
        self.clsc_shift_avd_prive = 0.0

        return
