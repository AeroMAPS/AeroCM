"""
Main interface to run climate simulations for aviation emissions using different climate models.
"""

import numpy as np
from copy import deepcopy
import xarray as xr
from collections.abc import Callable
from aerocm.climate_models.ipcc_climate_model import IPCCClimateModel
from aerocm.climate_models.gwpstar_climate_model import GWPStarClimateModel
from aerocm.climate_models.lwe_climate_model import LWEClimateModel
from aerocm.climate_models.fair_climate_model import FairClimateModel, FairRunner
from aerocm.utils.classes import ClimateModel


class AviationClimateSimulation:
    """
    Class to run a climate simulation for aviation emissions using a specified climate model.

    Parameters
    ----------
    climate_model : str | ClimateModel | Callable
        The climate model to use. Can be a registered name ('IPCC', 'GWP*', 'LWE', 'FaIR'), a subclass of ClimateModel, or a callable function
    start_year : int
        The starting year of the simulation.
    end_year : int
        The ending year of the simulation.
    species_inventory : dict
        A dictionary where keys are species names and values are arrays of emissions over time.
    species_settings : dict, optional
        A dictionary where keys are species names and values are dictionaries of settings for each species.
    model_settings : dict, optional
        A dictionary of settings for the climate model.

    Attributes
    ----------
    available_climate_models : list
        List of registered climate model names.

    Example
    -------
    ```python
    >>> import numpy as np
    >>> from aerocm.climate_models.aviation_climate_simulation import AviationClimateSimulation
    >>> from aerocm.utils.functions import plot_simulation_results
    >>> start_year = 2020
    >>> end_year = 2050
    >>> climate_model = "GWP*"
    >>> species_inventory = {
    ...     "CO2": np.random.rand(end_year - start_year + 1) * 1e9,  # in kg
    ...     "Contrails": np.random.rand(end_year - start_year + 1) * 1e9,  # in km
    ...     "NOx - ST O3": np.random.rand(end_year - start_year + 1) * 1e6,  # in kg
    ...     "NOx - CH4 and induced": np.random.rand(end_year - start_year + 1) * 1e6,  # in kg
    ...     "H2O": np.random.rand(end_year - start_year + 1) * 1e6,  # in kg
    ...     "Soot - ARI": np.random.rand(end_year - start_year + 1) * 1e6,  # in kg
    ...     "Sulfur - ARI": np.random.rand(end_year - start_year + 1) * 1e6,  # in kg
    ... }
    >>> species_settings = {
    ...     "CO2": {"ratio_erf_rf": 1.0},
    ...     "Contrails": {"sensitivity_rf": 2.23e-12, "ratio_erf_rf": 0.42, "efficacy_erf": 1.0},
    ...     "NOx - ST O3": {"sensitivity_rf": 7.6e-12, "ratio_erf_rf": 1.37, "efficacy_erf": 1.0},
    ...     "NOx - CH4 and induced": {"sensitivity_rf": -6.1e-12, "ratio_erf_rf": 1.18, "efficacy_erf": 1.0},
    ...     "H2O": {"sensitivity_rf": 5.2e-15, "ratio_erf_rf": 1.0, "efficacy_erf": 1.0},
    ...     "Soot - ARI": {"sensitivity_rf": 1.0e-10, "ratio_erf_rf": 1.0, "efficacy_erf": 1.0},
    ...     "Sulfur - ARI": {"sensitivity_rf": -2.0e-11, "ratio_erf_rf": 1.0, "efficacy_erf": 1.0},
    ... }
    >>> model_settings = {"tcre": 0.00045}
    >>> results = AviationClimateSimulation(
    ...     climate_model,
    ...     start_year,
    ...     end_year,
    ...     species_inventory,
    ...     species_settings,
    ...     model_settings
    ... ).run(return_xr=True)
    >>> plot_simulation_results(results, data_var="temperature_change", species=["CO2", "Non-CO2"], stacked=True)
    ```
    """

    # --- Variables for validation ---
    available_climate_models = ["IPCC", "GWP*", "LWE", "FaIR"]

    def __init__(
        self,
        climate_model: str | ClimateModel | Callable,
        start_year: int,
        end_year: int,
        species_inventory: dict,
        species_settings: dict | None = None,
        model_settings: dict | None = None,
    ):
        self.climate_model = climate_model
        self.start_year = start_year
        self.end_year = end_year
        self.species_inventory = species_inventory
        self.species_settings = species_settings or {}
        self.model_settings = model_settings or {}

        # --- Validate data ---
        self.validate_model()
        # Other checks (e.g. model and species settings) are done directly in the selected climate model

    def run(self, return_xr: bool = False) -> dict | xr.Dataset:
        """
        Run the climate simulation.

        Parameters
        ----------
        return_xr : bool
            If True, return results as an xarray Dataset. Default is False (returns a dictionary).

        Returns
        -------
        dict or xr.Dataset
            Results of the climate simulation.
        """

        # --- Extract species and their settings ---
        species_list = list(self.species_inventory.keys())
        species_settings = self.species_settings

        # --- Extract simulation parameters ---
        start_year = self.start_year
        end_year = self.end_year
        species_inventory = self.species_inventory.copy()
        years = list(range(start_year, end_year + 1))

        # --- Extract model and update species and model settings ---
        climate_model = self.climate_model
        model_settings = self.model_settings.copy()

        known_model = False
        if climate_model == "IPCC":
            climate_model = IPCCClimateModel
            known_model = True
        elif climate_model == "GWP*":
            climate_model = GWPStarClimateModel
            known_model = True
        elif climate_model == "LWE":
            climate_model = LWEClimateModel
            known_model = True
        elif climate_model == "FaIR":
            climate_model = FairClimateModel
            known_model = True

        if known_model:
            species_settings = add_default_species_settings(
                climate_model, species_settings
            )
            self.species_settings = (
                species_settings  # For accessing to the final species settings
            )
            model_settings = add_default_model_settings(climate_model, model_settings)
            self.model_settings = (
                model_settings  # For accessing to the final model settings
            )

        if climate_model == FairClimateModel and known_model:
            # -- Calculate background temperature and ERF only once here to improve calculation time ---
            background_species_quantities = (
                climate_model.get_background_species_quantities(
                    model_settings, start_year, end_year
                )
            )
            fair_runner = FairRunner(
                start_year, end_year, background_species_quantities
            )  # Initialize FairRunner
            results_background = (
                fair_runner.run()
            )  # Run with no additional species to get background state
            model_settings["background_temperature"] = results_background["temperature"]
            model_settings["background_effective_radiative_forcing"] = (
                results_background["effective_radiative_forcing"]
            )

        # -- Run model for all species ---
        results = {}
        if isinstance(climate_model, type) and issubclass(climate_model, ClimateModel):
            if "Contrails correction factors" in species_list:
                species_inventory["Contrails"] = (
                    species_inventory["Contrails"]
                    * species_inventory["Contrails correction factors"]
                )
                species_list.remove("Contrails correction factors")
            if "NOx" in species_list:
                species_list.append("NOx - ST O3")
                species_list.append("NOx - CH4 and induced")
                species_inventory["NOx - ST O3"] = species_inventory["NOx"]
                species_inventory["NOx - CH4 and induced"] = species_inventory["NOx"]
                species_list.remove("NOx")
            if "Soot" in species_list:
                species_list.append("Soot - ARI")
                species_list.append("Soot - ACI")
                species_inventory["Soot - ARI"] = species_inventory["Soot"]
                species_inventory["Soot - ACI"] = species_inventory["Soot"]
                species_list.remove("Soot")
            if "Sulfur" in species_list:
                species_list.append("Sulfur - ARI")
                species_list.append("Sulfur - ACI")
                species_inventory["Sulfur - ARI"] = species_inventory["Sulfur"]
                species_inventory["Sulfur - ACI"] = species_inventory["Sulfur"]
                species_list.remove("Sulfur")
            if "H2 leakage" in species_list:
                species_list.append("H2 leakage - ST O3")
                species_list.append("H2 leakage - CH4 and induced")
                species_inventory["H2 leakage - ST O3"] = species_inventory[
                    "H2 leakage"
                ]
                species_inventory["H2 leakage - CH4 and induced"] = species_inventory[
                    "H2 leakage"
                ]
                species_list.remove("H2 leakage")
            for specie in species_list:
                model_instance = climate_model(
                    start_year,
                    end_year,
                    specie,
                    species_inventory[specie],
                    species_settings[specie],
                    model_settings,
                )
                results[specie] = model_instance.run()
        elif callable(climate_model):
            for specie in species_list:
                results[specie] = climate_model(
                    start_year,
                    end_year,
                    specie,
                    species_inventory[specie],
                    species_settings[specie],
                    model_settings,
                )

        # --- NOX-CH4: discriminate between direct CH4 decrease and O3/H2O variations induced by CH4 decrease ---
        nox_ch4_results = results.get("NOx - CH4 and induced")
        if nox_ch4_results:
            f1 = 0.5  # Indirect effect of CH4 change on ozone
            f2 = 0.15  # Indirect effect of CH4 change on stratospheric water
            total_effect = 1 + f1 + f2

            factors = {
                "NOx - CH4": 1 / total_effect,
                "NOx - CH4 induced O3": f1 / total_effect,
                "NOx - CH4 induced H2O": f2 / total_effect,
            }

            for name, factor in factors.items():
                results[name] = {k: v * factor for k, v in nox_ch4_results.items()}

        # --- H2-CH4: discriminate between direct CH4 increase and O3/H2O variations induced by CH4 increase ---
        h2_leakage_ch4_results = results.get("H2 leakage - CH4 and induced")
        if h2_leakage_ch4_results:
            f1 = 0.5  # Indirect effect of CH4 change on ozone
            f2 = 0.15  # Indirect effect of CH4 change on stratospheric water
            total_effect = 1 + f1 + f2

            factors = {
                "H2 leakage - CH4": 1 / total_effect,
                "H2 leakage - CH4 induced O3": f1 / total_effect,
                "H2 leakage - CH4 induced H2O": f2 / total_effect,
            }

            for name, factor in factors.items():
                results[name] = {k: v * factor for k, v in h2_leakage_ch4_results.items()}

        # --- Aggregate results ---
        aggregations = {}
        if "NOx - CH4 and induced" in results:
            aggregations["NOx"] = ["NOx - ST O3", "NOx - CH4 and induced"]
        if "Soot - ARI" in results or "Soot - ACI" in results:
            aggregations["Soot"] = [
                s for s in ["Soot - ARI", "Soot - ACI"] if s in results
            ]
        if "Sulfur - ARI" in results or "Sulfur - ACI" in results:
            aggregations["Sulfur"] = [
                s for s in ["Sulfur - ARI", "Sulfur - ACI"] if s in results
            ]
        if (
            "Soot - ARI" in results
            or "Soot - ACI" in results
            or "Sulfur - ARI" in results
            or "Sulfur - ACI" in results
        ):
            aggregations["Aerosols"] = [
                s
                for s in ["Soot - ARI", "Soot - ACI", "Sulfur - ARI", "Sulfur - ACI"]
                if s in results
            ]
        if "H2 leakage - CH4 and induced" in results:
            aggregations["H2 leakage"] = [
                "H2 leakage - ST O3",
                "H2 leakage - CH4 and induced",
            ]
        if (
            "Contrails" in results
            or "NOx" in results
            or "H2O" in results
            or "Aerosols" in aggregations
        ):
            aggregations["Non-CO2"] = [
                s
                for s in ["Contrails", "NOx", "H2O", "Aerosols"]
                if s in results or s in aggregations
            ]
        if (
            "CO2" in results
            or "Non-CO2" in aggregations
            or "H2 leakage" in aggregations
        ):
            aggregations["Total"] = [
                s
                for s in ["CO2", "Non-CO2", "H2 leakage"]
                if s in results or s in aggregations
            ]

        for agg_name, names in aggregations.items():
            results[agg_name] = {
                key: sum(results[name][key] for name in names)
                for key in results[names[0]].keys()
            }

        # --- Convert to xarray if requested ---
        if return_xr:
            results = to_xarray(data=results, timesteps=years)

        return results

    def validate_model(self):
        """
        Validate the climate model provided.
        Specifically, check if it's a registered name, a subclass of ClimateModel, or a callable function.
        """
        model = self.climate_model

        is_registered_name = model in self.available_climate_models
        is_callable = callable(model)
        is_climate_subclass = isinstance(model, type) and issubclass(
            model, ClimateModel
        )

        if not (is_registered_name or is_callable or is_climate_subclass):
            raise ValueError(
                f"Climate model must be one of {self.available_climate_models}, "
                f"a subclass of ClimateModel, or a callable function"
            )


def to_xarray(data: dict, timesteps: list):
    """
    Convert results dictionary to xarray Dataset

    Parameters
    ----------
    data: dict
        dictionary with species as keys and another dictionary as values,
        where the inner dictionary has variable names (e.g. 'rf, 'erf') as keys and arrays as values
    timesteps: list
        Years corresponding to the data arrays
    """
    # Extract species and variable names
    species = list(data.keys())  # e.g. 'Aviation CO2', 'Aviation NOx', 'Aviation total'
    variables = sorted(
        {k for d in data.values() for k in d.keys()}
    )  # e.g. 'rf', 'erf', 'temperature increase'

    # Build xarray dataset
    ds = xr.Dataset(
        {
            var: (
                ("species", "year"),
                np.array(
                    [data[s].get(var, np.full(len(timesteps), np.nan)) for s in species]
                ),
            )
            for var in variables
        },
        coords={"species": species, "year": timesteps},
    )
    return ds


def add_default_species_settings(
    climate_model: ClimateModel, species_settings: dict | None = None
):
    """
    Complete the species settings with default values from the climate model for the ones that are not provided.

    Parameters
    ----------
    climate_model : ClimateModel
        The climate model class containing available species settings and their default values.
    species_settings : dict, optional
        The dictionary of species settings provided by the user. Defaults to None.

    Returns
    -------
    dict
        Updated species settings dictionary with default values filled in.
    """

    # Extract the default values from the available species settings of the climate model
    default_species_settings = {
        specie: {
            param: value["default"]
            for param, value in specie_param.items()
            if "default" in value
        }
        for specie, specie_param in climate_model.available_species_settings.items()
    }

    updated_species_settings = deepcopy(default_species_settings)

    # Update the values with the one provided by the user
    for specie, params in (species_settings or {}).items():
        if specie not in updated_species_settings:
            updated_species_settings[specie] = deepcopy(params)
        else:
            updated_species_settings[specie].update(params)

    return updated_species_settings


def add_default_model_settings(climate_model, model_settings):
    """
    Complete the model settings with default values from the climate model for the ones that are not provided.

    Parameters
    ----------
    climate_model : ClimateModel
        The climate model class containing available model settings and their default values.
    model_settings : dict, optional
        The dictionary of model settings provided by the user. Defaults to None.

    Returns
    -------
    dict
        Updated model settings dictionary with default values filled in.
    """

    # Extract the default values from the available model settings of the climate model
    default_model_settings = {
        key: value["default"]
        for key, value in climate_model.available_model_settings.items()
        if "default" in value
    }

    updated_model_settings = deepcopy(default_model_settings)

    # Update the values with the one provided by the user
    for key, value in (model_settings or {}).items():
        updated_model_settings[key] = value

    return updated_model_settings
