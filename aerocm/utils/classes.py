"""
Module containing the ClimateModel class for climate model implementations.
"""

import numpy as np
import logging
from abc import ABC, abstractmethod
import pandas as pd


class ClimateModel(ABC):
    """Super class for climate model implementations."""

    # --- Variables for model validation ---
    available_species = list()
    available_species_settings = dict()
    available_model_settings = dict()

    def __init__(
            self,
            start_year: int,
            end_year: int,
            specie_name: str,
            specie_inventory: list | np.ndarray,
            specie_settings: dict,
            model_settings: dict,
    ):
        """Initialize the climate model with the provided settings.

        Parameters
        ----------
        start_year : int
            Start year of the simulation.
        end_year : int
            End year of the simulation.
        specie_name : str
            Name of the species.
        specie_inventory : list or np.ndarray
            Emission profile for the species.
        specie_settings : dict
            Dictionary containing species settings.
        model_settings : dict
            Dictionary containing model settings.
        """

        # --- Validate parameters ---
        self.validate_model_settings(model_settings)
        self.validate_specie_settings(specie_name, specie_settings)
        self.validate_inventory(start_year, end_year, specie_inventory)

        # --- Store parameters ---
        self.start_year = start_year
        self.end_year = end_year
        self.specie_name = specie_name
        self.specie_inventory = specie_inventory
        self.specie_settings = specie_settings
        self.model_settings = model_settings

    @abstractmethod
    def run(self) -> dict | pd.DataFrame:
        """Run the climate model with the provided input data.

        Subclasses must return a dict with keys: 'radiative_forcing', 'effective_radiative_forcing',
        and 'temperature', which are the outputs of the climate model.
        Example:
            {
                'radiative_forcing': np.zeros(end_year - start_year + 1),
                'effective_radiative_forcing': np.zeros(end_year - start_year + 1),
                'temperature': np.zeros(end_year - start_year + 1)
            }
        """
        pass

    def validate_model_settings(self, model_settings: dict):
        """Validate the provided model settings.

        Parameters
        ----------
        model_settings : dict
            Dictionary containing model settings.

        Raises
        ------
        TypeError
            If any setting has an incorrect type.
        """
        for key in model_settings:
            if key in self.available_model_settings:
                if not isinstance(model_settings[key], self.available_model_settings[key]):
                    raise TypeError(f"Model setting {key} must be of type {self.available_model_settings[key]}")
            else:
                logging.info(f"Unknown model setting: {key}. Will be ignored.")

    def validate_specie_settings(self, specie_name: str, specie_settings: dict):
        """Validate the provided species settings.

        Parameters
        ----------
        specie_name : str
            Name of the species.
        specie_settings : dict
            Dictionary containing species settings.

        Raises
        ------
        ValueError
            If the species is not supported or if any mandatory setting is missing.
        TypeError
            If any setting has an incorrect type.
        """
        if specie_name not in self.available_species:
            raise ValueError(f"Species {specie_name} is not supported. Available species: {self.available_species}")
        available_specie_settings = self.available_species_settings[specie_name]
        for key in available_specie_settings:
            if key not in specie_settings:
                raise ValueError(f"Missing setting for {specie_name}: {key}")
            if not isinstance(specie_settings[key], available_specie_settings[key]):
                raise TypeError(f"Setting {key} for {specie_name} must be of type {available_specie_settings[key]}")
        for key in specie_settings:
            if key not in available_specie_settings:
                logging.info(f"Unknown setting for {specie_name}: {key}. Will be ignored.")

    def validate_inventory(self, start_year: int, end_year: int, specie_inventory: list | np.ndarray):
        """Validate the provided emission inventory.

        Parameters
        ----------
        start_year : int
            Start year of the simulation.
        end_year : int
            End year of the simulation.
        specie_inventory : list or np.ndarray
            Emission profile to validate.

        Raises
        ------
        ValueError
            If the emission profile length does not match the simulation period.
        """
        expected_length = end_year - start_year + 1
        if len(specie_inventory) != expected_length:
            raise ValueError(f"Inventory length must be {expected_length} for the period {start_year}-{end_year}")