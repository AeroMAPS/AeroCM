import numpy as np
from metrics.metrics import co2_ipcc_pulse_absolute_metrics


def species_lwe_climate_model(
    start_year, end_year, species_studied, species_quantities, params
):

    # params = {
    #     "sensitivity_erf": sensitivity_erf,
    #     "ratio_erf_rf": ratio_erf_rf,
    #     "efficacy_erf": efficacy_erf,
    #     "tcre": tcre
    # }

    sensitivity_erf = params["sensitivity_erf"]
    ratio_erf_rf = params["ratio_erf_rf"]
    efficacy_erf = params["efficacy_erf"]
    tcre = params["tcre"]

    if species_studied == "Aviation CO2":
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
        if species_studied == "Aviation NOx - CH4 decrease and induced":
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
        for i in range(0, size):
            for j in range(0, size):
                if i > j:
                    (
                        agwp_rf_co2_1,
                        agwp_erf_co2,
                        aegwp_rf_co2,
                        aegwp_erf_co2,
                        agtp_co2,
                        iagtp_co2,
                        atr_co2,
                    ) = co2_ipcc_pulse_absolute_metrics(i - j + 1)
                    (
                        agwp_rf_co2,
                        agwp_erf_co2,
                        aegwp_rf_co2,
                        aegwp_erf_co2,
                        agtp_co2,
                        iagtp_co2,
                        atr_co2,
                    ) = co2_ipcc_pulse_absolute_metrics(i - j)
                    F_co2[i, j] = agwp_rf_co2_1 - agwp_rf_co2
                elif i == j:
                    (
                        agwp_rf_co2,
                        agwp_erf_co2,
                        aegwp_rf_co2,
                        aegwp_erf_co2,
                        agtp_co2,
                        iagtp_co2,
                        atr_co2,
                    ) = co2_ipcc_pulse_absolute_metrics(1)
                    F_co2[i, j] = agwp_rf_co2

        F_co2_inv = np.linalg.inv(F_co2)
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
