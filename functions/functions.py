import numpy as np


def EmissionProfile(start_year, t0, time_horizon, profile, unit_value):
    emissions = np.zeros(t0-start_year+time_horizon+1)
    if profile == "pulse":
        emissions[t0-start_year] = unit_value
    elif profile == "step":
        for k in range(0, time_horizon+1):
            emissions[t0-start_year+k] = unit_value
    return emissions

def AbsoluteMetricsPulseDefaultCO2(time_horizon, unit_value):
    co2_molar_mass = 44.01 * 1e-3  # [kg/mol]
    air_molar_mass = 28.97e-3  # [kg/mol]
    atmosphere_total_mass = 5.1352e18  # [kg]
    radiative_efficiency = 1.37e-2 * 1e9  # radiative efficiency [mW/m^2]
    A_co2_unit = radiative_efficiency * air_molar_mass / (
                co2_molar_mass * atmosphere_total_mass) * 1e-3  # RF per unit mass increase in atmospheric abundance of CO2 [W/m^2/kg]
    A_co2 = A_co2_unit * unit_value
    a = [0.2173, 0.2240, 0.2824, 0.2763]
    tau = [0, 394.4, 36.54, 4.304]
    model_remaining_fraction_species_co2 = np.zeros(time_horizon+1)
    for k in range(0, time_horizon+1):
        model_remaining_fraction_species_co2[k] = a[0]
        for i in [1, 2, 3]:
            model_remaining_fraction_species_co2[k] += a[i] * np.exp(-k/tau[i])
    rf_co2 = A_co2 * model_remaining_fraction_species_co2
    agwp_co2 = A_co2 * a[0] * time_horizon
    for i in [1, 2, 3]:
        agwp_co2 += A_co2 * a[i] * tau[i] * (1 - np.exp(-time_horizon / tau[i]))
    aegwp_co2 = agwp_co2
    c = [0.631, 0.429]
    d = [8.4, 409.5]
    model_temperature_co2 = np.zeros(time_horizon+1)
    for k in range(0, time_horizon+1):
        for j in [0, 1]:
            term = a[0] * c[j] * (1 - np.exp(-k / d[j]))
            for i in [1, 2, 3]:
                term += a[i] * tau[i] * c[j] / (tau[i] - d[j]) * (np.exp(-k / tau[i]) - np.exp(-k / d[j]))
            model_temperature_co2[k] += A_co2 * term
    temp_co2 = model_temperature_co2
    iagtp_co2 = np.sum(model_temperature_co2)
    atr_co2 = 1 / time_horizon * iagtp_co2
    agtp_co2 = float(model_temperature_co2[-1])

    return rf_co2, agwp_co2, aegwp_co2, temp_co2, agtp_co2, iagtp_co2, atr_co2


def AbsoluteMetrics(radiative_forcing, effective_radiative_forcing, temperature, time_horizon):

    agwp = np.sum(radiative_forcing)
    aegwp = np.sum(effective_radiative_forcing)
    agtp = float(temperature[-1])
    iagtp = np.sum(temperature)
    atr = 1 / time_horizon * iagtp

    return agwp, aegwp, agtp, iagtp, atr


def RelativeMetrics(agwp_co2, aegwp_co2, agtp_co2, iagtp_co2, atr_co2, agwp, aegwp, agtp, iagtp, atr):

    gwp = agwp / agwp_co2
    egwp = aegwp / aegwp_co2
    gtp = agtp / agtp_co2
    igtp = iagtp / iagtp_co2
    ratr = atr / atr_co2

    return gwp, egwp, gtp, igtp, ratr