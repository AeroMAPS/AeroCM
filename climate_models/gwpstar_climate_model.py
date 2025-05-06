import numpy as np
from metrics.metrics import co2_ipcc_pulse_absolute_metrics


def gwpstar_equivalent_emissions_function(
    start_year,
    end_year,
    emissions_erf,
    gwpstar_variation_duration,
    gwpstar_s_coefficient,
):
    # Reference: Smith et al. (2021), https://doi.org/10.1038/s41612-021-00169-8
    # Global
    climate_time_horizon = 100
    (
        agwp_rf_co2,
        agwp_erf_co2,
        aegwp_rf_co2,
        aegwp_erf_co2,
        agtp_co2,
        iagtp_co2,
        atr_co2,
    ) = co2_ipcc_pulse_absolute_metrics(climate_time_horizon)
    co2_agwp_h = agwp_rf_co2

    # g coefficient for GWP*
    if gwpstar_s_coefficient == 0:
        g_coefficient = 1
    else:
        g_coefficient = (
            1 - np.exp(-gwpstar_s_coefficient / (1 - gwpstar_s_coefficient))
        ) / gwpstar_s_coefficient

    # Main
    emissions_erf_variation = np.zeros(end_year - start_year + 1)
    for k in range(start_year, end_year + 1):
        if k - start_year >= gwpstar_variation_duration:
            emissions_erf_variation[k - start_year] = (
                emissions_erf[k - start_year]
                - emissions_erf[k - gwpstar_variation_duration - start_year]
            ) / gwpstar_variation_duration
        else:
            emissions_erf_variation[k - start_year] = (
                emissions_erf[k - start_year] / gwpstar_variation_duration
            )
    emissions_equivalent_emissions = np.zeros(end_year - start_year + 1)
    for k in range(start_year, end_year + 1):
        emissions_equivalent_emissions[k - start_year] = (
            g_coefficient
            * (1 - gwpstar_s_coefficient)
            * climate_time_horizon
            / co2_agwp_h
            * emissions_erf_variation[k - start_year]
        ) + g_coefficient * gwpstar_s_coefficient / co2_agwp_h * emissions_erf[
            k - start_year
        ]

    return emissions_equivalent_emissions


def species_gwpstar_climate_model(
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
        effective_radiative_forcing = sensitivity_erf * species_quantities

        if (
            species_studied == "Aviation contrails"
            or species_studied == "Aviation NOx - ST O3 increase"
            or species_studied == "Aviation soot"
            or species_studied == "Aviation sulfur"
            or species_studied == "Aviation H2O"
        ):
            gwpstar_variation_duration = 6
            gwpstar_s_coefficient = 0.0

        elif species_studied == "Aviation NOx - CH4 decrease and induced":
            gwpstar_variation_duration = 20
            gwpstar_s_coefficient = 0.25

        equivalent_emissions = (
            gwpstar_equivalent_emissions_function(
                start_year,
                end_year,
                emissions_erf=effective_radiative_forcing,
                gwpstar_variation_duration=gwpstar_variation_duration,
                gwpstar_s_coefficient=gwpstar_s_coefficient,
            )
            / 10**12
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
