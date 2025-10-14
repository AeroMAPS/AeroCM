"""
Template for climate model implementations.
"""

import numpy as np
from typing import Union
import pandas as pd
from aerocm.utils.classes import ClimateModel


class MyClimateModel(ClimateModel):
    """ Template class for climate model implementations.

    Example usage
    -------------
    >>> import numpy as np
    >>> from aerocm.climate_models.template import MyClimateModel
    >>> start_year = 2020
    >>> end_year = 2050
    >>> specie_name = "species_1"
    >>> specie_inventory = np.random.rand(end_year - start_year + 1) * 1e9  # Example emission profile
    >>> specie_settings = {"param1": 1.0, "param2": 0.5}
    >>> model_settings = {"model_setting_1": np.array([0.1, 0.2, 0.3])}
    >>> climate_model = MyClimateModel(
    ...     start_year,
    ...     end_year,
    ...     specie_name,
    ...     specie_inventory,
    ...     specie_settings,
    ...     model_settings
    ... )
    >>> results = climate_model.run(return_df=True)
    """

    # --- Variables for validation ---
    available_species = [
        "species_1",
        "species_2",
        "species_3"
    ]
    available_species_settings = {
        "species_1": {"param1": float, "param2": float},
        "species_2": {"param3": int},
        "species_3": {},
    }
    available_model_settings = {"model_setting_1": Union[list, np.ndarray]}

    def run(self, return_df: bool = False) -> dict | pd.DataFrame:
        """Run the climate model with the assigned input data.

        Returns
        -------
        output_data : dict
            Dictionary containing the results of the climate model.
        """

        # --- Extract model settings ---
        model_setting_1 = self.model_settings["model_setting_1"]

        # --- Extract species settings ---
        specie_settings = self.specie_settings
        param1 = specie_settings.get("param1", 0.0)  # replace 2nd argument with default if needed
        param2 = specie_settings.get("param2", 1.0)
        param3 = specie_settings.get("param3", 1)

        # --- Run the climate model ---
        # Placeholder implementation - replace with actual model logic
        radiative_forcing = self.specie_inventory * param1
        effective_radiative_forcing = radiative_forcing * param2
        cumulative_effective_radiative_forcing = np.cumsum(effective_radiative_forcing)
        temperature = model_setting_1 * cumulative_effective_radiative_forcing * param3

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