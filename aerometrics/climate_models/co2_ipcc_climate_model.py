import numpy as np


def co2_ipcc_climate_model(start_year, end_year, unit_value):
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
    A_co2 = A_co2_unit * unit_value
    a = [0.2173, 0.2240, 0.2824, 0.2763]
    tau = [0, 394.4, 36.54, 4.304]
    model_remaining_fraction_species_co2 = np.zeros(end_year - start_year + 1)
    for k in range(0, end_year - start_year + 1):
        model_remaining_fraction_species_co2[k] = a[0]
        for i in [1, 2, 3]:
            model_remaining_fraction_species_co2[k] += a[i] * np.exp(-k / tau[i])
    radiative_forcing_co2 = A_co2 * model_remaining_fraction_species_co2
    effective_radiative_forcing_co2 = radiative_forcing_co2
    c = [0.631, 0.429]
    d = [8.4, 409.5]
    model_temperature_co2 = np.zeros(end_year - start_year + 1)
    for k in range(0, end_year - start_year + 1):
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
            model_temperature_co2[k] += A_co2 * term
    temperature_co2 = model_temperature_co2

    return radiative_forcing_co2, effective_radiative_forcing_co2, temperature_co2
