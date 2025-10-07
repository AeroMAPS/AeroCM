""" Module containing generic climate model functions for aviation species """
from typing import Union
import numpy as np
import warnings

from aerometrics.climate_models.gwpstar_climate_model import species_gwpstar_climate_model
from aerometrics.climate_models.lwe_climate_model import species_lwe_climate_model
from aerometrics.climate_models.fair_climate_model import (
    species_fair_climate_model,
    background_fair_climate_model,
)
from aerometrics.climate_models.constants import AVAILABLE_CLIMATE_MODELS, AVAILABLE_SPECIES, SPECIES_SETTINGS


def aviation_climate_model(
    start_year: int,
    end_year: int,
    climate_model: Union[str, callable],
    species_quantities: dict,
    species_settings: dict,
    model_settings: dict,
):
    """
    Generic climate model calculating RF, ERF and temperature increase for all CO2 and non-CO2 species
    from aviation emissions.
    :param start_year: start year of the assessment
    :param end_year: end year of the assessment
    :param climate_model: climate model. Must be one of AVAILABLE_CLIMATE_MODELS or a callable function
    :param species_quantities: dictionary {species: array of annual emissions/forcing values} with species as in AVAILABLE_SPECIES
    :param species_settings: dictionary {species: {species_setting: value}} with species as in AVAILABLE_SPECIES and species_setting as in SPECIES_SETTINGS
    :param model_settings: dictionary {model_setting: value}
    :return:
    """

    # --- Check inputs ---
    check_inputs(
        start_year,
        end_year,
        climate_model,
        species_quantities,
        species_settings,
        model_settings
    )

    # --- Prepare parameters ---
    params_model = model_settings.copy()
    if climate_model == "GWP*":
        climate_model = species_gwpstar_climate_model

    elif climate_model == "LWE":
        climate_model = species_lwe_climate_model

    elif climate_model == "FaIR":
        climate_model = species_fair_climate_model
        background_effective_radiative_forcing, background_temperature = (
            background_fair_climate_model(
                start_year, end_year, species_settings['Aviation CO2'], model_settings
            )
        )
        params_model["background_effective_radiative_forcing"] = (
            background_effective_radiative_forcing
        )
        params_model["background_temperature"] = background_temperature

    # --- Run model for all species ---
    results = {}
    for species in species_quantities.keys():
        rf, erf, temperature_increase = climate_model(
            start_year,
            end_year,
            species,
            species_quantities[species],
            species_settings[species],
            params_model,
        )
        results[species] = {
            "rf": rf,
            "erf": erf,
            "temperature increase": temperature_increase
        }
    # Species not included in the assessment are set to zero
    for species in AVAILABLE_SPECIES:
        if species not in results:
            warnings.warn(f"{species} not provided, setting contribution to zero")
            results[species] = {
                "rf": np.zeros(end_year - start_year + 1),
                "erf": np.zeros(end_year - start_year + 1),
                "temperature increase": np.zeros(end_year - start_year + 1)
            }

    # --- NOX-CH4 effects: discriminate between direct CH4 decrease and O3/H2O variations induced by CH4 decrease ---
    f1 = 0.5  # Indirect effect of CH4 decrease on ozone
    f2 = 0.15  # Indirect effect of CH4 decrease on stratospheric water
    results["Aviation NOX - CH4 decrease"] = {
        quantity: results["Aviation NOx - CH4 decrease and induced"][quantity] * (1 / (1 + f1 + f2)) for quantity in
        ["rf", "erf", "temperature increase"]
    }
    results["Aviation NOX - CH4 induced O3"] = {
        quantity: results["Aviation NOx - CH4 decrease and induced"][quantity] * (f1 / (1 + f1 + f2)) for quantity in
        ["rf", "erf", "temperature increase"]
    }
    results["Aviation NOX - CH4 induced H2O"] = {
        quantity: results["Aviation NOx - CH4 decrease and induced"][quantity] * (f2 / (1 + f1 + f2)) for quantity in
        ["rf", "erf", "temperature increase"]
    }

    # --- Aggregate intermediate results ---
    nox_keys = ["Aviation NOx - ST O3 increase", "Aviation NOx - CH4 decrease and induced"]
    results["Aviation NOx"] = {
        quantity: sum(results[key][quantity] for key in nox_keys)
        for quantity in ["rf", "erf", "temperature increase"]
    }

    aerosol_keys = ["Aviation soot", "Aviation sulfur"]
    results["Aviation aerosols"] = {
        quantity: sum(results[key][quantity] for key in aerosol_keys)
        for quantity in ["rf", "erf", "temperature increase"]
    }

    # --- Aggregate results ---
    non_co2_keys = ["Aviation contrails", "Aviation NOx", "Aviation H2O", "Aviation aerosols"]
    results["Aviation non-CO2"] = {
        quantity: sum(results[key][quantity] for key in non_co2_keys)
        for quantity in ["rf", "erf", "temperature increase"]
    }

    total_keys = ["Aviation CO2", "Aviation non-CO2"]
    results["Aviation total"] = {
        quantity: sum(results[key][quantity] for key in total_keys)
        for quantity in ["rf", "erf", "temperature increase"]
    }

    return results


def check_inputs(
        start_year: int,
        end_year: int,
        climate_model: Union[str, callable],
        species_quantities: dict,
        species_settings: dict,
        model_settings: dict
):
    # Check consistency of start and end year
    if end_year <= start_year:
        raise ValueError("end_year must be greater than start_year")

    # Check availability of climate model
    if climate_model not in AVAILABLE_CLIMATE_MODELS and not callable(climate_model):
        raise ValueError(
            f"Climate model must be one of {AVAILABLE_CLIMATE_MODELS} or a callable function"
        )

    # Check species_quantities is in the form {species: array of annual emissions/forcing values}
    for key, value in species_quantities.items():
        if key not in AVAILABLE_SPECIES:
            raise ValueError(f"species_quantities key {key} must be one of {AVAILABLE_SPECIES}")
        if not isinstance(value, (list, np.ndarray)):
            raise ValueError(f"species_quantities for {key} must be an array")
        if len(value) != end_year - start_year + 1:
            raise ValueError(f"species_quantities for {key} must have length {end_year - start_year + 1}")

    # Check species_settings is in the form {species: {species_setting: value}}
    for key, value in species_settings.items():
        if key not in AVAILABLE_SPECIES:
            raise ValueError(f"species_settings key {key} must be one of {AVAILABLE_SPECIES}")
        for setting in SPECIES_SETTINGS:
            if setting not in value:
                raise ValueError(f"species_settings for {key} must contain {setting}")
            if not isinstance(value[setting], float):
                raise ValueError(f"species_settings for {key} and {setting} must be a float")

    # Check consistency of species contained in species_quantities and species_settings
    if set(species_quantities.keys()) != set(species_settings.keys()):
        raise ValueError("species_quantities and species_settings must contain the same species")

    return
