import numpy as np
from scipy.linalg import solve_triangular
from scipy.interpolate import interp1d
from metrics.metrics import co2_ipcc_pulse_absolute_metrics


def species_lwe_climate_model(
    start_year, end_year, species, species_quantities, species_settings, model_settings
):

    # species_settings = {
    #     "sensitivity_erf": sensitivity_erf,
    #     "ratio_erf_rf": ratio_erf_rf,
    #     "efficacy_erf": efficacy_erf
    # }
    # model_settings = {
    #     "tcre": tcre
    # }

    sensitivity_erf = species_settings["sensitivity_erf"]
    ratio_erf_rf = species_settings["ratio_erf_rf"]
    efficacy_erf = species_settings["efficacy_erf"]
    tcre = model_settings["tcre"]

    if species == "Aviation CO2":
        equivalent_emissions = (
            species_quantities / 10**12
        )  # Conversion from kgCO2 to GtCO2

        co2_molar_mass = 44.01 * 1e-3  # [kg/mol]
        air_molar_mass = 28.97e-3  # [kg/mol]
        atmosphere_total_mass = 5.1352e18  # [kg]
        radiative_efficiency = 1.37e-2 * 1e9  # radiative efficiency [mW/m^2]
        A_co2_unit = (
            radiative_efficiency
            * air_molar_mass
            / (co2_molar_mass * atmosphere_total_mass)
            * 1e-3
        )  # RF per unit mass increase in atmospheric abundance of CO2 [W/m^2/kg]
        A_co2 = A_co2_unit * species_quantities
        a = [0.2173, 0.2240, 0.2824, 0.2763]
        tau = [0, 394.4, 36.54, 4.304]

        radiative_forcing_from_year = np.zeros(
            (len(species_quantities), len(species_quantities))
        )
        # Radiative forcing induced in year j by the species emitted in year i
        for i in range(0, len(species_quantities)):
            for j in range(0, len(species_quantities)):
                if i <= j:
                    radiative_forcing_from_year[i, j] = A_co2[i] * a[0]
                    for k in [1, 2, 3]:
                        radiative_forcing_from_year[i, j] += (
                            A_co2[i] * a[k] * np.exp(-(j - i) / tau[k])
                        )
        radiative_forcing = np.zeros(len(species_quantities))
        for k in range(0, len(species_quantities)):
            radiative_forcing[k] = np.sum(radiative_forcing_from_year[:, k])
        effective_radiative_forcing = radiative_forcing * ratio_erf_rf

    else:
        if species == "Aviation NOx - CH4 decrease and induced":
            tau_reference_year = [1940, 1980, 1994, 2004, 2050, 2300]
            tau_reference_values = [11, 10.1, 10, 9.85, 10.25, 10.25]
            tau_function = interp1d(
                tau_reference_year, tau_reference_values, kind="linear"
            )
            years = list(range(start_year, end_year + 1))
            tau = tau_function(years)
            A_CH4_unit = 5.7e-4
            A_CH4 = A_CH4_unit * sensitivity_erf * species_quantities
            f1 = 0.5  # Indirect effect on ozone
            f2 = 0.15  # Indirect effect on stratospheric water
            effective_radiative_forcing_from_year = np.zeros(
                (len(species_quantities), len(species_quantities))
            )
            # Radiative forcing induced in year j by the species emitted in year i
            for i in range(0, len(species_quantities)):
                for j in range(0, len(species_quantities)):
                    if i <= j:
                        effective_radiative_forcing_from_year[i, j] = (
                            (1 + f1 + f2) * A_CH4[i] * np.exp(-(j - i) / tau[j])
                        )
            effective_radiative_forcing = np.zeros(len(species_quantities))
            for k in range(0, len(species_quantities)):
                effective_radiative_forcing[k] = np.sum(effective_radiative_forcing_from_year[:, k])
            radiative_forcing = effective_radiative_forcing / ratio_erf_rf

        else:
            effective_radiative_forcing = sensitivity_erf * species_quantities
            radiative_forcing = effective_radiative_forcing / ratio_erf_rf

        size = end_year - start_year + 1
        F_co2 = np.zeros((size, size))

        # Old version for filling F_CO2 (long calculation time)
        # for i in range(0, size):
        #     for j in range(0, size):
        #         if i > j:
        #             start = time.time()
        #             agwp_rf_co2_1, *rest = co2_ipcc_pulse_absolute_metrics(i - j + 1)
        #             print("Matrix creation:", time.time() - start)
        #             agwp_rf_co2, *rest = co2_ipcc_pulse_absolute_metrics(i - j)
        #             F_co2[i, j] = agwp_rf_co2_1 - agwp_rf_co2
        #
        #         elif i == j:
        #             agwp_rf_co2, *rest = co2_ipcc_pulse_absolute_metrics(1)
        #             F_co2[i, j] = agwp_rf_co2

        agwp_data = {}
        for delta in range(1, size + 1):
            agwp, *rest = co2_ipcc_pulse_absolute_metrics(delta)
            agwp_data[delta] = agwp

        for i in range(size):
            for j in range(size):
                delta = i - j
                if delta > 0:
                    F_co2[i, j] = agwp_data[delta + 1] - agwp_data[delta]
                elif delta == 0:
                    F_co2[i, j] = agwp_data[1]

        # Inverting F_CO2 by using solve_triangular function (more efficient than np.linalg.inv)
        Identity = np.eye(F_co2.shape[0])
        F_co2_inv = solve_triangular(F_co2, Identity, lower=True)

        equivalent_emissions = (
            np.dot(F_co2_inv, effective_radiative_forcing) / 10**12
        )  # Conversion from kgCO2-we to GtCO2-we

    cumulative_equivalent_emissions = np.zeros(len(species_quantities))
    cumulative_equivalent_emissions[0] = equivalent_emissions[0]
    for k in range(1, len(cumulative_equivalent_emissions)):
        cumulative_equivalent_emissions[k] = (
            cumulative_equivalent_emissions[k - 1] + equivalent_emissions[k]
        )
    temperature = tcre * cumulative_equivalent_emissions * efficacy_erf

    return radiative_forcing, effective_radiative_forcing, temperature
