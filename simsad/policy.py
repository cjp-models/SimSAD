
class policy:
    """
    Paramètres du scénario

    Cette classe permet de choisir les paramètres du scénario simulé.

    Parameters
    ----------
    chsld_build: boolean
        True si on permet la construction de places en CHSLD (défaut=True)
    chsld_build_rate: float
        taux de construction de places en CHSLD durant une année 
        par rapport aux besoins de places dans ce milieu de vie (défaut=0.2)
    chsld_restricted_eligibility: boolean
        True si l'admissibilité au CHSLD est limitée pour les personnes avec un profil Iso-SMAF inférieur à 10 
        (défaut=False)
    chsld_restriction_rate: float
        taux de restriction des probabilités de transition vers les CHSLD pour les personnes 
        avec un profil Iso-SMAF inférieur à 10 (défaut=0.95)
    ri_build: boolean
        True si on permet la construction de places en RI-RTF (défaut=True)
    ri_build_rate: float
        taux de développement de places en RI-RTF durant une année 
        par rapport aux besoins de places dans ce milieu de vie (défaut=0.2)
    rpa_penetrate: boolean
        True si on permet le développement de nouvelles places en RPA financées par le public (défaut=False)
    rpa_penetrate_rate: float
        Proportion maximale de places en RPA financées par le public (défaut=0.25)
    rpa_adapt_rate: float
        taux de transformation des places de RPA en place subventionnée par rapport au nombre de personnes en attente 
        d'une place subventionnée en RPA (défaut=0.5)
    chsld_purchase: boolean
        True si on permet l'achat de places en CHSLD privé non-conventionné (défaut=True)
    chsld_purchase_rate: float
        taux d'achat supplémentaire de places en CHSLD non-conventionné durant une année 
        par rapport aux besoins de places en CHSLD (défaut=0.25)
    nsa_open_capacity: float
        proportion maximale de lit d'hôpitaux pouvant être occupés par les personnes en NSA (défaut=0.06)
    chsld_mda: boolean
        True si la construction des nouvelles places en CHSLD se fait selon le modèle des maisons des aînés 
        (défaut=True)
    infl_construction: float
        taux d'inflation des coûts de construction des CHSLD en dollar constant (défaut=0.01)
    interest_rate: float
        taux d'intérêt en dollar constant (défaut=0.03)
    clsc_cap: boolean
        True si limitation des heures de services fournis en CLSC selon la capacité en main d'oeuvre de ce fournisseur 
        (défaut=True)
    prive_cap: boolean
        True si limitation des heures de services fournis par le privé selon la capacité en main d'oeuvre 
        de ce fournisseur (défaut=True)
    eesad_cap: boolean
        True si limitation des heures de services fournis par les EÉSAD selon la capacité en main d'oeuvre 
        de ce fournisseur (défaut=True)
    purchase_prive: boolean
        True si achat d'heures de services fournis auprès du privé par les CLSC (défaut=True)
    purchase_eesad: boolean
        True si achat d'heures de services fournis auprès des EÉSAD par les CLSC (défaut=True)
    clsc_inf_rate: float
        taux d'ajustement de la main d'oeuvre par rapport aux besoins supplémentaire en main d'oeuvre
        pour les soins infirmiers en CSLC (défaut=0.25)
    clsc_avq_rate: float
        taux d'ajustement de la main d'oeuvre par rapport aux besoins supplémentaire en main d'oeuvre
        pour le soutien aux AVQ en CSLC (défaut=0.25)
    clsc_avd_rate: float
        taux d'ajustement de la main d'oeuvre par rapport aux besoins supplémentaire en main d'oeuvre
        pour le soutien aux AVD en CSLC (défaut=0.25)
    eesad_avd_rate: float
        taux d'ajustement de la main d'oeuvre par rapport aux besoins supplémentaire en main d'oeuvre
        pour le soutien aux AVD en EÉSAD (défaut=0.25)
    prive_avq_rate: float
        taux d'ajustement de la main d'oeuvre par rapport aux besoins supplémentaire en main d'oeuvre
        pour le soutien aux AVQ au privé (défaut=0.25)
    prive_avd_rate: float
        taux d'ajustement de la main d'oeuvre par rapport aux besoins supplémentaire en main d'oeuvre
        pour le soutien aux AVD au privé (défaut=0.25)
    chsld_inf_rate: float
        taux d'ajustement de la main d'oeuvre par rapport aux besoins supplémentaire en main d'oeuvre
        pour les soins infirmiers en CHSLD (défaut=0.25)
    chsld_avq_rate: float
        taux d'ajustement de la main d'oeuvre par rapport aux besoins supplémentaire en main d'oeuvre
        pour le soutien aux AVQ en CHSLD (défaut=0.25)
    ri_avq_rate: float
        taux d'ajustement de la main d'oeuvre par rapport aux besoins supplémentaire en main d'oeuvre
        pour le soutien aux AVQ en RI-RTF (défaut=0.25)
    ri_avd_rate: float
        taux d'ajustement de la main d'oeuvre par rapport aux besoins supplémentaire en main d'oeuvre
        pour le soutien aux AVD en RI-RTF (défaut=0.25)
    delta_inf_rate: float
        variation du taux de réponses aux besoins en soins infirmiers au SAD pour les personnes à domicile (défaut=0.0)
    delta_avq_rate: float
        variation du taux de réponses aux besoins en soutien aux AVQ au SAD pour les personnes à domicile (défaut=0.0)
    delta_avd_rate: float
        variation du taux de réponses aux besoins en soutien aux AVD au SAD pour les personnes à domicile (défaut=0.0)
    delta_cah_chsld: float
        variation du taux de contribution des usagers en CHSLD (défaut=0.0)
    delta_cah_ri: float
        variation du taux de contribution des usagers en RI-RTF (défaut=0.0)
    clsc_shift_avq_eesad: float
        proportion supplémentaire d'heures de services fournis en soutien aux AVQ par les CLSC transférées en achat 
        de services aux EÉSAD (défaut=0.0)
    clsc_shift_avq_prive: float
        proportion supplémentaire d'heures de services fournis en soutien aux AVQ par les CLSC transférées en achat 
        de services au privé (défaut=0.0)
    clsc_shift_avd_eesad: float
        proportion supplémentaire d'heures de services fournis en soutien aux AVD par les CLSC transférées en achat 
        de services aux EÉSAD (défaut=0.0)
    clsc_shift_avd_prive: float
        proportion supplémentaire d'heures de services fournis en soutien aux AVD par les CLSC transférées en achat 
        de services au privé (défaut=0.0)
    """
    def __init__(self):
        self.chsld_build = True
        self.chsld_build_rate = 0.2
        self.chsld_restricted_eligibility = False
        self.chsld_restriction_rate = 0.95
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
