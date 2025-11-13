""" Module containing generic climate metrics functions """
import warnings

import pandas as pd
from collections.abc import Callable
from aerocm.utils.classes import ClimateModel
from aerocm.climate_models.aviation_climate_simulation import AviationClimateSimulation
from aerocm.utils.functions import emission_profile_function
from aerocm.metrics.metrics import absolute_metrics, relative_metrics


class AviationClimateMetricsCalculation:
    """
    Class to calculate climate metrics for aviation emissions using a specified climate model.

    Example usage
    -------------
    >>> import numpy as np
    >>> from aerocm.metrics.aviation_climate_metrics_calculation import AviationClimateMetricsCalculation
    >>> climate_model = "FaIR"
    >>> start_year = 1940
    >>> time_horizon = [20, 50, 100]
    >>> species_profile = 'pulse'
    >>> profile_start_year = 2020
    >>> species_list = ["Contrails", "Soot"]
    >>> results = AviationClimateMetricsCalculation(
    ...     climate_model,
    ...     start_year,
    ...     time_horizon,
    ...     species_profile,
    ...     profile_start_year,
    ...     species_list
    ... ).run()
    """

    # --- Variables for validation ---
    available_climate_models = ['IPCC', 'GWP*', 'LWE', 'FaIR']

    available_species_profile = ['pulse', 'step', 'combined', 'scenario']

    def __init__(
            self,
            climate_model: str | ClimateModel | Callable,
            start_year: int,
            time_horizon: int | list,
            species_profile: str,
            profile_start_year: int | None = None,
            species_list: list = [],
            species_inventory: dict | None = None,
            species_settings: dict | None = None,
            model_settings: dict | None = None
    ):
        self.climate_model = climate_model
        self.start_year = start_year
        self.time_horizon = time_horizon
        self.species_profile = species_profile
        self.profile_start_year = profile_start_year
        self.species_list = species_list
        self.species_inventory = species_inventory
        self.species_settings = species_settings
        self.model_settings = model_settings

        # --- Validate data ---
        self.validate_model_profile()
        # Other checks (e.g. model and species settings) are done directly in the selected climate model

    def run(self, include_absolute_metrics: bool = False, return_df: bool = False) -> dict | pd.DataFrame:
        """
        Run the climate metric calculation.

        Returns
        -------
        dict
            Results of the climate metrics calculation.
        """

        # --- Extract simulation parameters ---
        climate_model = self.climate_model
        start_year = self.start_year
        time_horizon = self.time_horizon
        species_profile = self.species_profile
        profile_start_year = self.profile_start_year
        species_list = self.species_list
        species_inventory = self.species_inventory
        species_settings = self.species_settings
        model_settings = self.model_settings

        if climate_model == "FaIR":
            co2_unit_value = 1*10**10
            species_unit_value = {"Contrails": 1*10**10,
                                  "NOx - ST O3 increase": 1*10**10,
                                  "NOx - CH4 decrease and induced": 1*10**10,
                                  "H2O": 1*10**12,
                                  "Soot": 1*10**14,
                                  "Sulfur": 1*10**10
            }
        else:
            co2_unit_value = 1
            species_unit_value = {"Contrails": 1,
                                  "NOx - ST O3 increase": 1,
                                  "NOx - CH4 decrease and induced": 1,
                                  "H2O": 1,
                                  "Soot": 1,
                                  "Sulfur": 1
            }

        if type(time_horizon) == int or type(time_horizon) == float:
            time_horizon = [time_horizon]

        time_horizon_max = max(time_horizon)

        if species_profile == "pulse" or species_profile == "step":
            profile = species_profile
            co2_inventory = {
                "CO2": emission_profile_function(start_year,
                                                 profile_start_year,
                                                 time_horizon_max,
                                                 profile=profile,
                                                 unit_value=co2_unit_value
                                                 )
            }
            non_co2_inventory = {
                specie: emission_profile_function(start_year,
                                                  profile_start_year,
                                                  time_horizon_max,
                                                  profile=profile,
                                                  unit_value=species_unit_value[specie]
                                                  )
                for specie in species_list
            }
        elif species_profile == "combined":
            co2_inventory = {
                "CO2": emission_profile_function(start_year,
                                                 profile_start_year,
                                                 time_horizon_max,
                                                 profile="pulse",
                                                 unit_value=co2_unit_value
                                                 )
            }
            non_co2_inventory = {
                specie: emission_profile_function(start_year,
                                                  profile_start_year,
                                                  time_horizon_max,
                                                  profile="step",
                                                  unit_value=species_unit_value[specie]
                                                  )
                for specie in species_list
            }
        elif species_profile == "scenario":
            co2_inventory = {"CO2": species_inventory["CO2"]}
            non_co2_inventory = {specie: params for specie, params in species_inventory.items() if specie != 'CO2'}
            species_list = [k for k in species_inventory.keys() if k != "CO2"]

        if species_profile != "scenario":
            end_year = profile_start_year + time_horizon_max
        else:
            first_key = next(iter(species_inventory))
            first_value = species_inventory[first_key]
            size = len(first_value)
            end_year = size + start_year - 1

        # -- Run model for CO2 ---
        full_co2_climate_simulation_results = AviationClimateSimulation(
            climate_model=climate_model,
            start_year=start_year,
            end_year=end_year,
            species_inventory=co2_inventory,
            species_settings=species_settings,
            model_settings=model_settings).run()

        # -- Run model for all species ---
        full_non_co2_climate_simulation = AviationClimateSimulation(
            climate_model=climate_model,
            start_year=start_year,
            end_year=end_year,
            species_inventory=non_co2_inventory,
            species_settings=species_settings,
            model_settings=model_settings)
        full_non_co2_climate_simulation_results = full_non_co2_climate_simulation.run()
        non_co2_species_settings = full_non_co2_climate_simulation.species_settings

        # -- Remove useless data and divide by unit values --
        co2_climate_simulation_results = {
            "CO2": {key: value / co2_unit_value
                    for key, value in full_co2_climate_simulation_results["CO2"].items()}
        }
        non_co2_climate_simulation_results = {
            specie: {key: value / species_unit_value[specie]
                    for key, value in full_non_co2_climate_simulation_results[specie].items()}
            for specie in species_list
        }

        # --- Calculate metrics for each time horizon ---
        results = {}

        for H in time_horizon:

            # --- Absolute metrics ---
            results_H_absolute = {}

            # CO2
            agwp_rf_co2, agwp_erf_co2, aegwp_rf_co2, aegwp_erf_co2, agtp_co2, iagtp_co2, atr_co2 = absolute_metrics(
                co2_climate_simulation_results["CO2"]["radiative_forcing"][
                :end_year - start_year + 1 - (time_horizon_max - H)],
                co2_climate_simulation_results["CO2"]["effective_radiative_forcing"][
                :end_year - start_year + 1 - (time_horizon_max - H)],
                1.0,
                co2_climate_simulation_results["CO2"]["temperature"][
                :end_year - start_year + 1 - (time_horizon_max - H)],
                H
            )

            results_H_absolute["CO2"] = {
                "agwp_rf": agwp_rf_co2,
                "agwp_erf": agwp_erf_co2,
                "aegwp_rf": aegwp_rf_co2,
                "aegwp_erf": aegwp_erf_co2,
                "agtp": agtp_co2,
                "iagtp": iagtp_co2,
                "atr": atr_co2
            }

            # Non-CO2 species
            for specie in species_list:
                agwp_rf, agwp_erf, aegwp_rf, aegwp_erf, agtp, iagtp, atr = absolute_metrics(
                    non_co2_climate_simulation_results[specie]["radiative_forcing"][
                    :end_year - start_year + 1 - (time_horizon_max - H)],
                    non_co2_climate_simulation_results[specie]["effective_radiative_forcing"][
                    :end_year - start_year + 1 - (time_horizon_max - H)],
                    non_co2_species_settings[specie]["efficacy_erf"],
                    non_co2_climate_simulation_results[specie]["temperature"][
                    :end_year - start_year + 1 - (time_horizon_max - H)],
                    H
                )

                results_H_absolute[specie] = {
                    "agwp_rf": agwp_rf,
                    "agwp_erf": agwp_erf,
                    "aegwp_rf": aegwp_rf,
                    "aegwp_erf": aegwp_erf,
                    "agtp": agtp,
                    "iagtp": iagtp,
                    "atr": atr
                }

            # --- Relative metrics ---
            results_H_relative = {}

            for specie in species_list + ["CO2"]:
                gwp_rf, gwp_erf, egwp_rf, egwp_erf, gtp, igtp, ratr = relative_metrics(
                    results_H_absolute["CO2"]["agwp_rf"],
                    results_H_absolute["CO2"]["agwp_erf"],
                    results_H_absolute["CO2"]["aegwp_rf"],
                    results_H_absolute["CO2"]["aegwp_erf"],
                    results_H_absolute["CO2"]["agtp"],
                    results_H_absolute["CO2"]["iagtp"],
                    results_H_absolute["CO2"]["atr"],
                    results_H_absolute[specie]["agwp_rf"],
                    results_H_absolute[specie]["agwp_erf"],
                    results_H_absolute[specie]["aegwp_rf"],
                    results_H_absolute[specie]["aegwp_erf"],
                    results_H_absolute[specie]["agtp"],
                    results_H_absolute[specie]["iagtp"],
                    results_H_absolute[specie]["atr"]
                )

                results_H_relative[specie] = {
                    "gwp_rf": gwp_rf,
                    "gwp_erf": gwp_erf,
                    "egwp_rf": egwp_rf,
                    "egwp_erf": egwp_erf,
                    "gtp": gtp,
                    "igtp": igtp,
                    "ratr": ratr
                }

            if include_absolute_metrics:
                results[H] = {
                    specie: {
                        **results_H_absolute[specie],
                        **results_H_relative[specie]
                    }
                    for specie in species_list + ["CO2"]
                }
            else:
                results[H] = {
                    specie: {
                        **results_H_relative[specie]
                    }
                    for specie in species_list  # Exclude CO2 if only relative metrics are requested (values are 1.0)
                }

        if return_df:
            flatten_dicts = []
            for H, species_dict in results.items():
                for specie, metrics in species_dict.items():
                    flatten_dict = {"time_horizon": H, "species": specie, **{k: float(v) for k, v in metrics.items()}}
                    flatten_dicts.append(flatten_dict)

            results = pd.DataFrame(flatten_dicts)

        return results

    def validate_model_profile(self):
        model = self.climate_model
        species_profile = self.species_profile

        if model == "GWP*":
            warnings.warn(f"The '{model}' climate model is not recommended for calculating aviation climate metrics.")

        is_registered_name = species_profile in self.available_species_profile

        if not is_registered_name:
            raise ValueError(
                f"Species profile must be one of {self.available_species_profile}"
                )
