""" Module containing generic climate metrics functions """
import numpy as np
import xarray as xr
import warnings
from collections.abc import Callable
from aerocm.climate_models.gwpstar_climate_model import GWPStarClimateModel
from aerocm.climate_models.lwe_climate_model import LWEClimateModel
from aerocm.climate_models.fair_climate_model import FairClimateModel, FairRunner
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
    >>> start_year = 1940
    >>> end_year = 2050
    >>> climate_model = "FaIR"
    >>> species_inventory = {
    ...     "CO2": np.random.rand(end_year - start_year + 1) * 1e9,  # in kg
    ...     "Contrails": np.random.rand(end_year - start_year + 1) * 1e-3,  # in W/m^2
    ...     "NOx - ST O3 increase": np.random.rand(end_year - start_year + 1) * 1e6,  # in kg
    ...     "NOx - CH4 decrease and induced": np.random.rand(end_year - start_year + 1) * 1e6,  # in kg
    ...     "H2O": np.random.rand(end_year - start_year + 1) * 1e6,  # in kg
    ...     "Soot": np.random.rand(end_year - start_year + 1) * 1e6,  # in kg
    ...     "Sulfur": np.random.rand(end_year - start_year + 1) * 1e6,  # in kg
    ... }
    >>> species_settings = {
    ...     "CO2": {"sensitivity_rf": 1.0, "ratio_erf_rf": 1.0, "efficacy_erf": 1.0},
    ...     "Contrails": {"sensitivity_rf": 2.23e-12, "ratio_erf_rf": 0.42, "efficacy_erf": 1.0},
    ...     "NOx - ST O3 increase": {"sensitivity_rf": 7.6e-12, "ratio_erf_rf": 1.37, "efficacy_erf": 1.0},
    ...     "NOx - CH4 decrease and induced": {"sensitivity_rf": -6.1e-12, "ratio_erf_rf": 1.18, "efficacy_erf": 1.0},
    ...     "H2O": {"sensitivity_rf": 5.2e-15, "ratio_erf_rf": 1.0, "efficacy_erf": 1.0},
    ...     "Soot": {"sensitivity_rf": 1.0e-10, "ratio_erf_rf": 1.0, "efficacy_erf": 1.0},
    ...     "Sulfur": {"sensitivity_rf": -2.0e-11, "ratio_erf_rf": 1.0, "efficacy_erf": 1.0},
    ... }
    >>> model_settings = {"tcre": 0.00045}
    >>> results = AviationClimateMetricsCalculation(
    ...     climate_model,
    ...     start_year,
    ...     end_year,
    ...     species_inventory,
    ...     species_settings,
    ...     model_settings
    ... ).run(return_xr=True)
    >>> plot_simulation_results(results, data_var="temperature_change", species=["CO2", "Non-CO2"], stacked=True)
    """

    # --- Variables for validation ---
    available_climate_models = ['GWP*', 'LWE', 'FaIR']

    available_species_profile = ['pulse', 'step', 'combined', 'scenario']

    def __init__(
            self,
            climate_model: str | ClimateModel | Callable,
            start_year: int,
            metrics_list: list,
            time_horizon: int | float | list,
            species_profile: dict,
            profile_start_year: int | None = None,
            species_list: list | None = None,
            species_inventory: dict | None = None,
            species_settings: dict | None = None,
            model_settings: dict | None = None
    ):
        self.climate_model = climate_model
        self.start_year = start_year
        self.metrics_list = metrics_list
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

    def run(self, return_xr: bool = False) -> dict | xr.Dataset:
        """
        Run the climate metric calculation.

       Parameters
        ----------
        return_xr : bool
            If True, return results as an xarray Dataset. Default is False (returns a dictionary).

        Returns
        -------
        dict or xr.Dataset
            Results of the climate simulation.
        """

        # --- Extract simulation parameters ---
        climate_model = self.climate_model
        start_year = self.start_year
        metrics_list = self.metrics_list
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
            co2_inventory = species_inventory["CO2"]
            non_co2_inventory = {specie: params for specie, params in species_inventory.items()}
            species_list = list(species_inventory.keys())

        if species_profile != "scenario":
            end_year = profile_start_year + time_horizon_max
        else:
            first_key = next(iter(species_inventory))
            first_value = species_inventory[first_key]
            size = len(first_value)
            end_year = size + start_year - 1

        # -- Run model for CO2 ---
        full_co2_climate_simulation_results = {}
        full_co2_climate_simulation_results = AviationClimateSimulation(
            climate_model=climate_model,
            start_year=start_year,
            end_year=end_year,
            species_inventory=co2_inventory,
            species_settings=species_settings,
            model_settings=model_settings).run()

        # -- Run model for all species ---
        full_non_co2_climate_simulation_results = {}
        full_non_co2_climate_simulation_results = AviationClimateSimulation(
            climate_model=climate_model,
            start_year=start_year,
            end_year=end_year,
            species_inventory=non_co2_inventory,
            species_settings=species_settings,
            model_settings=model_settings).run()

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

        # -- Calculate absolute metrics ---
        absolute_metrics_results = []
        for H in time_horizon:
            absolute_metrics_results_H = {}
            # CO2
            agwp_rf_co2, agwp_erf_co2, aegwp_rf_co2, aegwp_erf_co2, agtp_co2, iagtp_co2, atr_co2 = absolute_metrics(
                co2_climate_simulation_results["CO2"]["radiative_forcing"],
                co2_climate_simulation_results["CO2"]["effective_radiative_forcing"],
                1.0,
                co2_climate_simulation_results["CO2"]["temperature"],
                H)
            absolute_metrics_results_H["CO2"] = {"agwp_rf": agwp_rf_co2,
                                                 "agwp_erf": agwp_erf_co2,
                                                 "aegwp_rf": aegwp_rf_co2,
                                                 "aegwp_erf": aegwp_erf_co2,
                                                 "agtp": agtp_co2,
                                                 "iagtp": iagtp_co2,
                                                 "atr": atr_co2}
            # Species
            for specie in species_list:
                agwp_rf, agwp_erf, aegwp_rf, aegwp_erf, agtp, iagtp, atr = absolute_metrics(
                    non_co2_climate_simulation_results[specie]["radiative_forcing"],
                    non_co2_climate_simulation_results[specie]["effective_radiative_forcing"],
                    1.0, # TODO
                    non_co2_climate_simulation_results[specie]["temperature"],
                    H)
                absolute_metrics_results_H[specie] = {"agwp_rf": agwp_rf,
                                                     "agwp_erf": agwp_erf,
                                                     "aegwp_rf": aegwp_rf,
                                                     "aegwp_erf": aegwp_erf,
                                                     "agtp": agtp,
                                                     "iagtp": iagtp,
                                                     "atr": atr}
            absolute_metrics_results += [absolute_metrics_results_H]

        # -- Calculate relative metrics ---
        relative_metrics_results = []
        print(time_horizon)
        for k in range(0, len(time_horizon)):
            relative_metrics_results_H = {}
            print(absolute_metrics_results[k]["CO2"]["agwp_rf"])
            for specie in species_list:
                gwp_rf, gwp_erf, egwp_rf, egwp_erf, gtp, igtp, ratr = relative_metrics(
                    absolute_metrics_results[k]["CO2"]["agwp_rf"],
                    absolute_metrics_results[k]["CO2"]["agwp_erf"],
                    absolute_metrics_results[k]["CO2"]["aegwp_rf"],
                    absolute_metrics_results[k]["CO2"]["aegwp_erf"],
                    absolute_metrics_results[k]["CO2"]["agtp"],
                    absolute_metrics_results[k]["CO2"]["iagtp"],
                    absolute_metrics_results[k]["CO2"]["atr"],
                    absolute_metrics_results[k][specie]["agwp_rf"],
                    absolute_metrics_results[k][specie]["agwp_erf"],
                    absolute_metrics_results[k][specie]["aegwp_rf"],
                    absolute_metrics_results[k][specie]["aegwp_erf"],
                    absolute_metrics_results[k][specie]["agtp"],
                    absolute_metrics_results[k][specie]["iagtp"],
                    absolute_metrics_results[k][specie]["atr"])
                relative_metrics_results_H[specie] = {"gwp_rf": gwp_rf,
                                                      "gwp_erf": gwp_erf,
                                                      "egwp_rf": egwp_rf,
                                                      "egwp_erf": egwp_erf,
                                                      "gtp": gtp,
                                                      "igtp": igtp,
                                                      "ratr": ratr}

            relative_metrics_results += [relative_metrics_results_H]

        return absolute_metrics_results, relative_metrics_results


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
