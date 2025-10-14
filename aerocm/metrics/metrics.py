import numpy as np


def co2_ipcc_pulse_absolute_metrics(time_horizon):
    co2_molar_mass = 44.01 * 1e-3  # [kg/mol]
    air_molar_mass = 28.97e-3  # [kg/mol]
    atmosphere_total_mass = 5.1352e18  # [kg]
    radiative_efficiency = 1.33e-5  # radiative efficiency [W/m^2/ppb] with AR6 value
    A_co2_unit = (
            radiative_efficiency
            * 1e9
            * air_molar_mass
            / (co2_molar_mass * atmosphere_total_mass)
    )  # RF per unit mass increase in atmospheric abundance of CO2 [W/m^2/kg]
    a = [0.2173, 0.2240, 0.2824, 0.2763]
    tau = [0, 394.4, 36.54, 4.304]
    model_remaining_fraction_species_co2 = np.zeros(time_horizon + 1)
    for k in range(0, time_horizon + 1):
        model_remaining_fraction_species_co2[k] = a[0]
        for i in [1, 2, 3]:
            model_remaining_fraction_species_co2[k] += a[i] * np.exp(-k / tau[i])
    agwp_co2 = A_co2_unit * a[0] * time_horizon
    for i in [1, 2, 3]:
        agwp_co2 += A_co2_unit * a[i] * tau[i] * (1 - np.exp(-time_horizon / tau[i]))
    agwp_rf_co2 = agwp_co2
    agwp_erf_co2 = agwp_co2
    aegwp_rf_co2 = agwp_co2
    aegwp_erf_co2 = agwp_co2
    c = [0.631, 0.429]
    d = [8.4, 409.5]
    model_temperature_co2 = np.zeros(time_horizon + 1)
    for k in range(0, time_horizon + 1):
        for j in [0, 1]:
            term = a[0] * c[j] * (1 - np.exp(-k / d[j]))
            for i in [1, 2, 3]:
                term += (
                    a[i]
                    * tau[i]
                    * c[j]
                    / (tau[i] - d[j])
                    * (np.exp(-k / tau[i]) - np.exp(-k / d[j]))
                )
            model_temperature_co2[k] += A_co2_unit * term
    iagtp_co2 = np.sum(model_temperature_co2)
    atr_co2 = 1 / time_horizon * iagtp_co2
    agtp_co2 = float(model_temperature_co2[-1])

    return (
        agwp_rf_co2,
        agwp_erf_co2,
        aegwp_rf_co2,
        aegwp_erf_co2,
        agtp_co2,
        iagtp_co2,
        atr_co2,
    )


def absolute_metrics(
    radiative_forcing,
    effective_radiative_forcing,
    efficacy_erf,
    temperature,
    time_horizon,
):

    agwp_rf = np.sum(radiative_forcing)
    agwp_erf = np.sum(effective_radiative_forcing)
    efficacy_rf = efficacy_erf * agwp_erf / agwp_rf
    aegwp_rf = efficacy_rf * np.sum(radiative_forcing)
    aegwp_erf = efficacy_erf * np.sum(effective_radiative_forcing)
    agtp = float(temperature[-1])
    iagtp = np.sum(temperature)
    atr = 1 / time_horizon * iagtp

    return agwp_rf, agwp_erf, aegwp_rf, aegwp_erf, agtp, iagtp, atr


def relative_metrics(
    agwp_rf_co2,
    agwp_erf_co2,
    aegwp_rf_co2,
    aegwp_erf_co2,
    agtp_co2,
    iagtp_co2,
    atr_co2,
    agwp_rf,
    agwp_erf,
    aegwp_rf,
    aegwp_erf,
    agtp,
    iagtp,
    atr,
):

    gwp_rf = agwp_rf / agwp_rf_co2
    gwp_erf = agwp_erf / agwp_erf_co2
    egwp_rf = aegwp_rf / aegwp_rf_co2
    egwp_erf = aegwp_erf / aegwp_erf_co2
    gtp = agtp / agtp_co2
    igtp = iagtp / iagtp_co2
    ratr = atr / atr_co2

    return gwp_rf, gwp_erf, egwp_rf, egwp_erf, gtp, igtp, ratr
