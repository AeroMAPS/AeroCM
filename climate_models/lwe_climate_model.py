import numpy as np
from scipy.linalg import solve_triangular
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
        effective_radiative_forcing = np.zeros(len(species_quantities))
        _, agwp_erf_100_co2, *rest = co2_ipcc_pulse_absolute_metrics(100)
        sensitivity_erf = agwp_erf_100_co2 / 100
        effective_radiative_forcing[0] = sensitivity_erf * species_quantities[0]
        for k in range(1, len(species_quantities)):
            effective_radiative_forcing[k] = (
                effective_radiative_forcing[k - 1]
                + sensitivity_erf * species_quantities[k]
            )

    else:
        if species == "Aviation NOx - CH4 decrease and induced":
            tau = 11.8
            A_CH4_unit = 5.7e-4
            A_CH4 = A_CH4_unit * sensitivity_erf * species_quantities
            f1 = 0.5  # Indirect effect on ozone
            f2 = 0.15  # Indirect effect on stratospheric water
            effective_radiative_forcing_from_year = np.zeros(
                (len(species_quantities), len(species_quantities))
            )
            # Effective radiative forcing induced in year j by the species emitted in year i
            for i in range(0, len(species_quantities)):
                for j in range(0, len(species_quantities)):
                    if i <= j:
                        effective_radiative_forcing_from_year[i, j] = (
                            (1 + f1 + f2) * A_CH4[i] * np.exp(-(j - i) / tau)
                        )
            effective_radiative_forcing = np.zeros(len(species_quantities))
            for k in range(0, len(species_quantities)):
                effective_radiative_forcing[k] = np.sum(
                    effective_radiative_forcing_from_year[:, k]
                )

        else:
            effective_radiative_forcing = sensitivity_erf * species_quantities

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

    radiative_forcing = effective_radiative_forcing / ratio_erf_rf
    cumulative_equivalent_emissions = np.zeros(len(species_quantities))
    cumulative_equivalent_emissions[0] = equivalent_emissions[0]
    for k in range(1, len(cumulative_equivalent_emissions)):
        cumulative_equivalent_emissions[k] = (
            cumulative_equivalent_emissions[k - 1] + equivalent_emissions[k]
        )
    temperature = tcre * cumulative_equivalent_emissions * efficacy_erf

    return radiative_forcing, effective_radiative_forcing, temperature
