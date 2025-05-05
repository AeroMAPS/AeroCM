import numpy as np


def aeromaps_climate_model(
    start_year, end_year, species_climate_model_function, species_quantities, params
):

    params_species = {}

    if (
        species_climate_model_function == "SpeciesLWEClimateModel"
        or species_climate_model_function == "SpeciesGWPStarClimateModel"
    ):
        sensitivity_erf = params["sensitivity_erf"]
        ratio_erf_rf = params["ratio_erf_rf"]
        efficacy_erf = params["efficacy_erf"]
        tcre = params["tcre"]
        for k in range(0, 7):
            params_species[k] = {
                "sensitivity_erf": sensitivity_erf[k],
                "ratio_erf_rf": ratio_erf_rf[k],
                "efficacy_erf": efficacy_erf[k],
                "tcre": tcre,
            }

    elif species_climate_model_function == "SpeciesFaIRClimateModel":
        background_species_quantities = params["background_species_quantities"]
        sensitivity_erf = params["sensitivity_erf"]
        ratio_erf_rf = params["ratio_erf_rf"]
        efficacy_erf = params["efficacy_erf"]
        for k in range(0, 7):
            params_species[k] = {
                "background_species_quantities": background_species_quantities,
                "sensitivity_erf": sensitivity_erf[k],
                "ratio_erf_rf": ratio_erf_rf[k],
                "efficacy_erf": efficacy_erf[k],
            }

    else:
        print("The chosen climate model is not available in AeroMAPS.")

    # CO2
    co2_rf, co2_erf, temperature_increase_from_co2_from_aviation = (
        species_climate_model_function(
            start_year,
            end_year,
            "Aviation CO2",
            species_quantities[0],
            params_species[0],
        )
    )

    # Contrails
    contrails_rf, contrails_erf, temperature_increase_from_contrails_from_aviation = (
        species_climate_model_function(
            start_year,
            end_year,
            "Aviation contrails",
            species_quantities[1],
            params_species[1],
        )
    )

    # NOx - ST O3 increase
    nox_st_o3_rf, nox_st_o3_erf, temperature_increase_from_nox_st_o3_from_aviation = (
        species_climate_model_function(
            start_year,
            end_year,
            "Aviation NOx - ST O3 increase",
            species_quantities[2],
            params_species[2],
        )
    )

    # NOx - CH4 decrease and induced
    nox_ch4_rf, nox_ch4_erf, temperature_increase_from_nox_ch4_from_aviation = (
        species_climate_model_function(
            start_year,
            end_year,
            "Aviation NOx - CH4 decrease and induced",
            species_quantities[3],
            params_species[3],
        )
    )
    nox_rf = nox_st_o3_rf + nox_ch4_rf
    nox_erf = nox_st_o3_erf + nox_ch4_erf
    temperature_increase_from_nox_from_aviation = (
        temperature_increase_from_nox_st_o3_from_aviation
        + temperature_increase_from_nox_ch4_from_aviation
    )
    f1 = 0.5  # Indirect effect on ozone
    f2 = 0.15  # Indirect effect on stratospheric water
    nox_ch4_decrease_rf = nox_ch4_rf * (1 / (1 + f1 + f2))
    nox_ch4_decrease_erf = nox_ch4_erf * (1 / (1 + f1 + f2))
    temperature_increase_from_nox_ch4_decrease_from_aviation = (
        temperature_increase_from_nox_ch4_from_aviation * (1 / (1 + f1 + f2))
    )
    nox_ch4_o3_rf = nox_ch4_rf * (f1 / (1 + f1 + f2))
    nox_ch4_o3_erf = nox_ch4_erf * (f1 / (1 + f1 + f2))
    temperature_increase_from_nox_ch4_o3_from_aviation = (
        temperature_increase_from_nox_ch4_from_aviation * (f1 / (1 + f1 + f2))
    )
    nox_ch4_h2o_rf = nox_ch4_rf * (f2 / (1 + f1 + f2))
    nox_ch4_h2o_erf = nox_ch4_erf * (f2 / (1 + f1 + f2))
    temperature_increase_from_nox_ch4_h2o_from_aviation = (
        temperature_increase_from_nox_ch4_from_aviation * (f2 / (1 + f1 + f2))
    )

    # H2O
    h2o_rf, h2o_erf, temperature_increase_from_h2o_from_aviation = (
        species_climate_model_function(
            start_year,
            end_year,
            "Aviation H2O",
            species_quantities[4],
            params_species[4],
        )
    )

    # Soot
    soot_rf, soot_erf, temperature_increase_from_soot_from_aviation = (
        species_climate_model_function(
            start_year,
            end_year,
            "Aviation soot",
            species_quantities[5],
            params_species[5],
        )
    )

    # Soot
    sulfur_rf, sulfur_erf, temperature_increase_from_sulfur_from_aviation = (
        species_climate_model_function(
            start_year,
            end_year,
            "Aviation sulfur",
            species_quantities[6],
            params_species[6],
        )
    )
    aerosols_rf = soot_rf + sulfur_rf
    aerosols_erf = soot_erf + sulfur_erf
    temperature_increase_from_aerosols_from_aviation = (
        temperature_increase_from_soot_from_aviation
        + temperature_increase_from_sulfur_from_aviation
    )

    # Total
    non_co2_rf = contrails_rf + nox_rf + h2o_rf + soot_rf + sulfur_rf
    non_co2_erf = contrails_erf + nox_erf + h2o_erf + soot_erf + sulfur_erf
    temperature_increase_from_non_co2_from_aviation = (
        temperature_increase_from_contrails_from_aviation
        + temperature_increase_from_nox_from_aviation
        + temperature_increase_from_h2o_from_aviation
        + temperature_increase_from_soot_from_aviation
        + temperature_increase_from_sulfur_from_aviation
    )
    total_rf = co2_rf + non_co2_rf
    total_erf = co2_erf + non_co2_erf
    temperature_increase_from_aviation = (
        temperature_increase_from_co2_from_aviation
        + temperature_increase_from_non_co2_from_aviation
    )

    return (
        co2_rf,
        co2_erf,
        temperature_increase_from_co2_from_aviation,
        contrails_rf,
        contrails_erf,
        temperature_increase_from_contrails_from_aviation,
        nox_rf,
        nox_erf,
        temperature_increase_from_nox_from_aviation,
        nox_st_o3_rf,
        nox_st_o3_erf,
        temperature_increase_from_nox_st_o3_from_aviation,
        nox_ch4_rf,
        nox_ch4_erf,
        temperature_increase_from_nox_ch4_from_aviation,
        nox_ch4_decrease_rf,
        nox_ch4_decrease_erf,
        temperature_increase_from_nox_ch4_decrease_from_aviation,
        nox_ch4_o3_rf,
        nox_ch4_o3_erf,
        temperature_increase_from_nox_ch4_o3_from_aviation,
        nox_ch4_h2o_rf,
        nox_ch4_h2o_erf,
        temperature_increase_from_nox_ch4_h2o_from_aviation,
        h2o_rf,
        h2o_erf,
        temperature_increase_from_h2o_from_aviation,
        aerosols_rf,
        aerosols_erf,
        temperature_increase_from_aerosols_from_aviation,
        soot_rf,
        soot_erf,
        temperature_increase_from_soot_from_aviation,
        sulfur_rf,
        sulfur_erf,
        temperature_increase_from_sulfur_from_aviation,
        non_co2_rf,
        non_co2_erf,
        temperature_increase_from_non_co2_from_aviation,
        total_rf,
        total_erf,
        temperature_increase_from_aviation,
    )
