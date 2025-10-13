"""
Module containing the ClimateModel class for climate model implementations.
"""

import numpy as np
from typing import Union
import logging
from abc import ABC, abstractmethod
import pandas as pd


class ClimateModel(ABC):
    """Super class for climate model implementations."""

    # --- Variables for model validation ---
    available_species = dict()
    available_species_settings = dict()
    mandatory_model_settings = dict()
    optional_model_settings = dict()

    def __init__(
            self,
            start_year: int,
            end_year: int,
            species: str,
            emission_profile: Union[list, np.ndarray],
            species_settings: dict,
            model_settings: dict,
    ):
        """Initialize the climate model with the provided settings.

        Parameters
        ----------
        model_settings : dict
            Dictionary containing model settings.
        start_year : int
            Start year of the simulation.
        end_year : int
            End year of the simulation.
        species : str
            Name of the species.
        emission_profile : list or np.ndarray
            Emission profile for the species.
        species_settings : dict
            Dictionary containing species settings.
        model_settings : dict
            Dictionary containing model settings.
        """

        # --- Validate parameters ---
        self.validate_model_settings(model_settings)
        self.validate_species_settings(species, species_settings)
        self.validate_emission_profile(start_year, end_year, emission_profile)

        # --- Store parameters ---
        self.start_year = start_year
        self.end_year = end_year
        self.species = species
        self.emission_profile = emission_profile
        self.species_settings = species_settings
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
        ValueError
            If any mandatory setting is missing.
        TypeError
            If any setting has an incorrect type.
        """
        for key in self.mandatory_model_settings:
            if key not in model_settings:
                raise ValueError(f"Missing mandatory model setting: {key}")
            if not isinstance(model_settings[key], self.mandatory_model_settings[key]):
                raise TypeError(f"Model setting {key} must be of type {self.mandatory_model_settings[key]}")
        for key in self.optional_model_settings:
            if key in model_settings and not isinstance(model_settings[key], self.optional_model_settings[key]):
                raise TypeError(f"Model setting {key} must be of type {self.optional_model_settings[key]}")
        for key in model_settings:
            if key not in self.mandatory_model_settings and key not in self.optional_model_settings:
                logging.info(f"Unknown model setting: {key}. Will be ignored.")

    def validate_species_settings(self, species: str, species_settings: dict):
        """Validate the provided species settings.

        Parameters
        ----------
        species : str
            Name of the species.
        species_settings : dict
            Dictionary containing species settings.

        Raises
        ------
        ValueError
            If the species is not supported or if any mandatory setting is missing.
        TypeError
            If any setting has an incorrect type.
        """
        if species not in self.available_species:
            raise ValueError(f"Species {species} is not supported. Available species: {self.available_species}")
        mandatory_settings = self.available_species_settings[species]
        for key in mandatory_settings:
            if key not in species_settings:
                raise ValueError(f"Missing mandatory setting for {species}: {key}")
            if not isinstance(species_settings[key], mandatory_settings[key]):
                raise TypeError(f"Setting {key} for {species} must be of type {mandatory_settings[key]}")
        for key in species_settings:
            if key not in mandatory_settings:
                logging.info(f"Unknown setting for {species}: {key}. Will be ignored.")

    def validate_emission_profile(self, start_year: int, end_year: int, emission_profile: Union[list, np.ndarray]):
        """Validate the provided emission profile.

        Parameters
        ----------
        start_year : int
            Start year of the simulation.
        end_year : int
            End year of the simulation.
        emission_profile : list or np.ndarray
            Emission profile to validate.

        Raises
        ------
        ValueError
            If the emission profile length does not match the simulation period.
        """
        expected_length = end_year - start_year + 1
        if len(emission_profile) != expected_length:
            raise ValueError(f"Emission profile length must be {expected_length} for the period {start_year}-{end_year}")