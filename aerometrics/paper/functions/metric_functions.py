import numpy as np

def gwp_rf_contrails(time_horizon, sensitivity_rf_contrails, ratio_erf_rf_contrails, efficacy_erf_contrails):
    metric = 4.55e14 * sensitivity_rf_contrails / time_horizon**0.801
    return metric

def gwp_erf_contrails(time_horizon, sensitivity_rf_contrails, ratio_erf_rf_contrails, efficacy_erf_contrails):
    metric = 4.55e14 * sensitivity_rf_contrails * ratio_erf_rf_contrails / time_horizon**0.801
    return metric

def egwp_contrails(time_horizon, sensitivity_rf_contrails, ratio_erf_rf_contrails, efficacy_erf_contrails):
    metric = 4.55e14 * sensitivity_rf_contrails * ratio_erf_rf_contrails * efficacy_erf_contrails / time_horizon**0.801
    return metric
    
def ratr_contrails(time_horizon, sensitivity_rf_contrails, ratio_erf_rf_contrails, efficacy_erf_contrails):
    metric = 6.21e14 * sensitivity_rf_contrails * ratio_erf_rf_contrails * efficacy_erf_contrails / time_horizon**0.842
    return metric

def ratr_contrails(time_horizon, sensitivity_rf_contrails, ratio_erf_rf_contrails, efficacy_erf_contrails):
    metric = 6.21e14 * sensitivity_rf_contrails * ratio_erf_rf_contrails * efficacy_erf_contrails / time_horizon**0.842
    return metric

def ratr_nox_o3(time_horizon, sensitivity_rf_nox_o3, ratio_erf_rf_nox_o3, efficacy_erf_nox_o3):
    metric = 6.21e14 * sensitivity_rf_nox_o3 * ratio_erf_rf_nox_o3 * efficacy_erf_nox_o3 / time_horizon**0.842
    return metric

def ratr_nox_ch4(time_horizon, sensitivity_rf_nox_ch4, ratio_erf_rf_nox_ch4, efficacy_erf_nox_ch4):
    metric = 138 * sensitivity_rf_nox_ch4 * ratio_erf_rf_nox_ch4 * efficacy_erf_nox_ch4 * time_horizon**(0.140 - 0.106 * np.log(time_horizon))
    return metric