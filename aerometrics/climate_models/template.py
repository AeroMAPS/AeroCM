"""
Template for climate model implementations.
"""

import numpy as np
from typing import Union

import pandas as pd

from aerometrics.utils.classes import ClimateModel


class MyClimateModel(ClimateModel):
    """ Template class for climate model implementations."""

    # --- Variables for validation ---
    available_species = {
        "CO2",
        "Contrails",
        "NOx - ST O3 increase",
        "NOx - CH4 decrease and induced",
        "Soot",
        "Sulfur",
        "H2O"
    }
    available_species_settings = {
        "CO2": {"ratio_erf_rf": float, "efficacy_erf": float},
        "Contrails": {"sensitivity_rf": float, "ratio_erf_rf": float, "efficacy_erf": float},
        "NOx - ST O3 increase": {"sensitivity_rf": float, "ratio_erf_rf": float, "efficacy_erf": float},
        "NOx - CH4 decrease and induced": {"sensitivity_rf": float, "ratio_erf_rf": float, "efficacy_erf": float},
        "Soot": {"sensitivity_rf": float, "ratio_erf_rf": float, "efficacy_erf": float},
        "Sulfur": {"sensitivity_rf": float, "ratio_erf_rf": float, "efficacy_erf": float},
        "H2O": {"sensitivity_rf": float, "ratio_erf_rf": float, "efficacy_erf": float},
    }
    mandatory_model_settings = {"tcre": float}
    optional_model_settings = {"my_optional_setting": Union[list, np.ndarray]}

    def __init__(
            self,
            start_year: int,
            end_year: int,
            species: str,
            emission_profile: Union[list, np.ndarray],
            species_settings: dict,
            model_settings: dict,
            any_other_parameter=None  # Example of an additional parameter
    ):
        """ Modification of __init__ method is only needed if additional parameters are required by the model."""

        # --- Call the super class constructor for validations and assignments ---
        super().__init__(start_year, end_year, species, emission_profile, species_settings, model_settings)

        # --- Initialize model-specific variables here if needed ---
        self.any_other_parameter = any_other_parameter

    def run(self, return_df : bool = False) -> dict | pd.DataFrame:
        """Run the climate model with the assigned input data.

        Returns
        -------
        output_data : dict
            Dictionary containing the results of the climate model.
        """

        # --- Extract model settings ---
        tcre = self.model_settings["tcre"]

        # --- Extract species settings ---
        species_settings = self.species_settings
        sensitivity_rf = species_settings.get("sensitivity_rf", 0.0)  # replace 2nd argument with default if needed
        ratio_erf_rf = species_settings.get("ratio_erf_rf", 1.0)
        efficacy_erf = species_settings.get("efficacy_erf", 1.0)

        # --- Run the climate model ---
        # Placeholder implementation - replace with actual model logic
        radiative_forcing = self.emission_profile * sensitivity_rf
        effective_radiative_forcing = radiative_forcing * ratio_erf_rf
        cumulative_effective_radiative_forcing = np.cumsum(effective_radiative_forcing)
        temperature = tcre * cumulative_effective_radiative_forcing * efficacy_erf

        # --- Prepare output data ---
        output_data = {
            "radiative_forcing": radiative_forcing,
            "effective_radiative_forcing": effective_radiative_forcing,
            "temperature": temperature
        }

        if return_df:
            years = np.arange(self.start_year, self.end_year + 1)
            output_data = pd.DataFrame(output_data, index=years)

        return output_data