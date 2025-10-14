import numpy as np

def gwp_rf_contrails(time_horizon, sensitivity_rf_contrails, ratio_erf_rf_contrails, efficacy_erf_contrails):
    metric = 4.71e14 * sensitivity_rf_contrails / time_horizon**0.830
    return metric

def gwp_erf_contrails(time_horizon, sensitivity_rf_contrails, ratio_erf_rf_contrails, efficacy_erf_contrails):
    metric = 4.71e14 * sensitivity_rf_contrails * ratio_erf_rf_contrails / time_horizon**0.830
    return metric

def egwp_contrails(time_horizon, sensitivity_rf_contrails, ratio_erf_rf_contrails, efficacy_erf_contrails):
    metric = 4.71e14 * sensitivity_rf_contrails * ratio_erf_rf_contrails * efficacy_erf_contrails / time_horizon**0.830
    return metric
    
def ratr_contrails(time_horizon, sensitivity_rf_contrails, ratio_erf_rf_contrails, efficacy_erf_contrails):
    metric = 7.20e14 * sensitivity_rf_contrails * ratio_erf_rf_contrails * efficacy_erf_contrails / time_horizon**0.880
    return metric

def ratr_nox_o3(time_horizon, sensitivity_rf_nox_o3, ratio_erf_rf_nox_o3, efficacy_erf_nox_o3):
    metric = 7.20e14 * sensitivity_rf_nox_o3 * ratio_erf_rf_nox_o3 * efficacy_erf_nox_o3 / time_horizon**0.880
    return metric

def ratr_nox_ch4(time_horizon, ch4_loss_per_nox, ratio_erf_rf_nox_ch4, efficacy_erf_nox_ch4):
    metric = 128.9 * ch4_loss_per_nox * ratio_erf_rf_nox_ch4 * efficacy_erf_nox_ch4 * time_horizon**(0.185 - 0.114 * np.log(time_horizon))
    return metric