from aerometrics.climate_models.gwpstar_climate_model import species_gwpstar_climate_model
from aerometrics.climate_models.lwe_climate_model import species_lwe_climate_model
from aerometrics.climate_models.fair_climate_model import (
    species_fair_climate_model,
    background_fair_climate_model,
)


def aviation_climate_model(
    start_year,
    end_year,
    species_climate_model,
    species_quantities,
    species_settings,
    model_settings,
):

    params_species = {}
    sensitivity_rf = species_settings["sensitivity_rf"]
    ratio_erf_rf = species_settings["ratio_erf_rf"]
    efficacy_erf = species_settings["efficacy_erf"]
    for k in range(0, 7):
        params_species[k] = {
            "sensitivity_rf": sensitivity_rf[k],
            "ratio_erf_rf": ratio_erf_rf[k],
            "efficacy_erf": efficacy_erf[k],
        }

    params_model = model_settings.copy()

    if species_climate_model == "GWP*":
        species_climate_model = species_gwpstar_climate_model

    elif species_climate_model == "LWE":
        species_climate_model = species_lwe_climate_model

    elif species_climate_model == "FaIR":
        species_climate_model = species_fair_climate_model
        background_effective_radiative_forcing, background_temperature = (
            background_fair_climate_model(
                start_year, end_year, params_species[0], model_settings
            )
        )
        params_model["background_effective_radiative_forcing"] = (
            background_effective_radiative_forcing
        )
        params_model["background_temperature"] = background_temperature

    else:
        print(
            "The chosen climate model is directly provided or not available in AeroMAPS."
        )

    # CO2
    co2_rf, co2_erf, temperature_increase_from_co2_from_aviation = (
        species_climate_model(
            start_year,
            end_year,
            "Aviation CO2",
            species_quantities[0],
            params_species[0],
            params_model,
        )
    )

    # Contrails
    contrails_rf, contrails_erf, temperature_increase_from_contrails_from_aviation = (
        species_climate_model(
            start_year,
            end_year,
            "Aviation contrails",
            species_quantities[1],
            params_species[1],
            params_model,
        )
    )

    # NOx - ST O3 increase
    nox_st_o3_rf, nox_st_o3_erf, temperature_increase_from_nox_st_o3_from_aviation = (
        species_climate_model(
            start_year,
            end_year,
            "Aviation NOx - ST O3 increase",
            species_quantities[2],
            params_species[2],
            params_model,
        )
    )

    # NOx - CH4 decrease and induced
    nox_ch4_rf, nox_ch4_erf, temperature_increase_from_nox_ch4_from_aviation = (
        species_climate_model(
            start_year,
            end_year,
            "Aviation NOx - CH4 decrease and induced",
            species_quantities[3],
            params_species[3],
            params_model,
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
        species_climate_model(
            start_year,
            end_year,
            "Aviation H2O",
            species_quantities[4],
            params_species[4],
            params_model,
        )
    )

    # Soot
    soot_rf, soot_erf, temperature_increase_from_soot_from_aviation = (
        species_climate_model(
            start_year,
            end_year,
            "Aviation soot",
            species_quantities[5],
            params_species[5],
            params_model,
        )
    )

    # Soot
    sulfur_rf, sulfur_erf, temperature_increase_from_sulfur_from_aviation = (
        species_climate_model(
            start_year,
            end_year,
            "Aviation sulfur",
            species_quantities[6],
            params_species[6],
            params_model,
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
