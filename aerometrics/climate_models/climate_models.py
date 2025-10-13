""" Module containing generic climate model functions for aviation species """
from typing import Union
import numpy as np
import xarray as xr

from aerometrics.climate_models.gwpstar_climate_model import GWPStarClimateModel
from aerometrics.climate_models.lwe_climate_model import LWEClimateModel
from aerometrics.climate_models.fair_climate_model import FairClimateModel
from aerometrics.utils.classes import ClimateModel


class AviationClimateSimulation:
    """
    Class to run a climate simulation for aviation emissions using a specified climate model.

    Example usage
    -------------
    >>> import numpy as np
    >>> from aerometrics.climate_models.climate_models import AviationClimateSimulation
    >>> from aerometrics.utils.functions import plot_simulation_results
    >>> start_year = 2020
    >>> end_year = 2050
    >>> climate_model = "GWP*"
    >>> emission_profiles = {
    ...     "CO2": np.random.rand(end_year - start_year + 1) * 1e9,  # in kg
    ...     "NOx - ST O3 increase": np.random.rand(end_year - start_year + 1) * 1e6,  # in kg
    ...     "NOx - CH4 decrease and induced": np.random.rand(end_year - start_year + 1) * 1e6,  # in kg
    ...     "Contrails": np.random.rand(end_year - start_year + 1) * 1e-3,  # in W/m^2
    ...     "H2O": np.random.rand(end_year - start_year + 1) * 1e6,  # in kg
    ...     "Soot": np.random.rand(end_year - start_year + 1) * 1e6,  # in kg
    ...     "Sulfur": np.random.rand(end_year - start_year + 1) * 1e6,  # in kg
    ... }
    >>> species_settings = {
    ...     "CO2": {"sensitivity_rf": 1.0, "ratio_erf_rf": 1.0, "efficacy_erf": 1.0},
    ...     "NOx - ST O3 increase": {"sensitivity_rf": 7.6e-12, "ratio_erf_rf": 1.37, "efficacy_erf": 1.0},
    ...     "NOx - CH4 decrease and induced": {"sensitivity_rf": -6.1e-12, "ratio_erf_rf": 1.18, "efficacy_erf": 1.0},
    ...     "Contrails": {"sensitivity_rf": 2.23e-12, "ratio_erf_rf": 0.42, "efficacy_erf": 1.0},
    ...     "H2O": {"sensitivity_rf": 5.2e-15, "ratio_erf_rf": 1.0, "efficacy_erf": 1.0},
    ...     "Soot": {"sensitivity_rf": 1.0e-10, "ratio_erf_rf": 1.0, "efficacy_erf": 1.0},
    ...     "Sulfur": {"sensitivity_rf": -2.0e-11, "ratio_erf_rf": 1.0, "efficacy_erf": 1.0},
    ... }
    >>> model_settings = {"tcre": 0.00045}
    >>> results = AviationClimateSimulation(
    ...     climate_model,
    ...     start_year,
    ...     end_year,
    ...     emission_profiles,
    ...     species_settings,
    ...     model_settings
    ... ).run(return_xr=True)
    >>> plot_simulation_results(results, data_var="temperature_change", species=["CO2", "Non-CO2"], stacked=True)
    """

    # --- Variables for validation ---
    available_climate_models = ['GWP*', 'LWE', 'FaIR']

    def __init__(
            self,
            climate_model: Union[str, ClimateModel, callable],
            start_year: int,
            end_year: int,
            emission_profiles: dict,
            species_settings: dict,
            model_settings: dict
    ):
        self.climate_model = climate_model
        self.start_year = start_year
        self.end_year = end_year
        self.emission_profiles = emission_profiles
        self.species_settings = species_settings
        self.model_settings = model_settings

        # --- Validate data ---
        self.validate_model()
        # Other checks (e.g. model and species settings) are done directly in the selected climate model

    def run(self, return_xr: bool = False) -> Union[dict, xr.Dataset]:
        """
        Run the climate simulation.

       Parameters
        ----------
        return_xr : bool
            If True, return results as an xarray Dataset. Default is False (returns a dictionary).

        Returns
        -------
        Union[dict, xr.Dataset]
            Results of the climate simulation.
        """

        # --- Extract model and its settings ---
        climate_model = self.climate_model
        model_settings = self.model_settings
        if climate_model == "GWP*":
            climate_model = GWPStarClimateModel
        elif climate_model == "LWE":
            climate_model = LWEClimateModel
        elif climate_model == "FaIR":
            climate_model = FairClimateModel

        # --- Extract species and their settings ---
        species_list = list(self.emission_profiles.keys())
        species_settings = self.species_settings

        # --- Extract simulation parameters ---
        start_year = self.start_year
        end_year = self.end_year
        emission_profiles = self.emission_profiles
        years = list(range(start_year, end_year + 1))

        # -- Run model for all species ---
        results = {}
        for specie in species_list:
            if isinstance(climate_model, type) and issubclass(climate_model, ClimateModel):
                model_instance = climate_model(
                    start_year,
                    end_year,
                    specie,
                    emission_profiles[specie],
                    species_settings[specie],
                    model_settings,
                )
                results[specie] = model_instance.run()
            elif callable(climate_model):
                results[specie] = climate_model(
                    start_year,
                    end_year,
                    specie,
                    emission_profiles[specie],
                    species_settings[specie],
                    model_settings,
                )

        # --- NOX-CH4: discriminate between direct CH4 decrease and O3/H2O variations induced by CH4 decrease ---
        nox_ch4_results = results.get("NOx - CH4 decrease and induced")
        if nox_ch4_results:
            f1 = 0.5  # Indirect effect of CH4 decrease on ozone
            f2 = 0.15  # Indirect effect of CH4 decrease on stratospheric water
            total_effect = 1 + f1 + f2

            factors = {
                "NOX - CH4 decrease": 1 / total_effect,
                "NOX - CH4 induced O3": f1 / total_effect,
                "NOX - CH4 induced H2O": f2 / total_effect,
            }

            for name, factor in factors.items():
                results[name] = {k: v * factor for k, v in nox_ch4_results.items()}

        # --- Aggregate results ---
        aggregations = {}
        if "NOx - CH4 decrease and induced" in results:
            aggregations["NOx"] = ["NOx - ST O3 increase", "NOx - CH4 decrease and induced"]
        if "Soot" in results or "Sulfur" in results:
            aggregations["Aerosols"] = [s for s in ["Soot", "Sulfur"] if s in results]
        if "Contrails" in results or "NOx" in results or "H2O" in results or "Aerosols" in aggregations:
            aggregations["Non-CO2"] = [s for s in ["Contrails", "NOx", "H2O", "Aerosols"] if s in results or s in aggregations]
        if "CO2" in results or "Non-CO2" in aggregations:
            aggregations["Total"] = [s for s in ["CO2", "Non-CO2"] if s in results or s in aggregations]

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
        model = self.climate_model

        is_registered_name = model in self.available_climate_models
        is_callable = callable(model)
        is_climate_subclass = isinstance(model, type) and issubclass(model, ClimateModel)

        if not (is_registered_name or is_callable or is_climate_subclass):
            raise ValueError(
                f"Climate model must be one of {self.available_climate_models}, "
                f"a subclass of ClimateModel, or a callable function"
            )


def to_xarray(data: dict, timesteps: list):
    """
    Convert results dictionary to xarray Dataset
    :param data: dictionary {species: {variable: array of values}}
    :param years: list of years corresponding to the time dimension
    """
    # Extract species and variable names
    species = list(data.keys())  # e.g. 'Aviation CO2', 'Aviation NOx', 'Aviation total'
    variables = sorted({k for d in data.values() for k in d.keys()})  # e.g. 'rf', 'erf', 'temperature increase'

    # Build xarray dataset
    ds = xr.Dataset(
        {
            var: (("species", "year"),
                  np.array([data[s].get(var, np.full(len(timesteps), np.nan)) for s in species]))
            for var in variables
        },
        coords={
            "species": species,
            "year": timesteps
        }
    )
    return ds




