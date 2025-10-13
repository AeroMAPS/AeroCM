import warnings

from aerometrics.climate_data import RCP
import os.path as pth
import numpy as np
import pandas as pd
from fair import FAIR
from fair.interface import fill, initialise
from scipy.interpolate import interp1d
from typing import Union
from aerometrics.utils.classes import ClimateModel

RCP_START_YEAR = 1765


class FairRunner:
    """
    Class to run FaIR climate model for a (single) given species and its emission profile.

    ---
    Example usage:
    >>> import numpy as np
    >>> from aerometrics.climate_models.fair_climate_model import FairRunner
    >>> from aerometrics.climate_models.fair_climate_model import background_species_quantities_function
    >>> start_year = 2020
    >>> end_year = 2050
    >>> species = "CO2"
    >>> species_quantities = np.array([1e12] * (end_year - start_year + 1))  # 1 GtCO2 per year
    >>> species_settings = {
    ...     "sensitivity_rf": 1.0,
    ...     "ratio_erf_rf": 1.0,
    ...     "efficacy_erf": 1.0,
    ... }
    >>> background_species_quantities = background_species_quantities_function(start_year, end_year, 'RCP45')
    >>> fair_runner = FairRunner(start_year, end_year, background_species_quantities)
    >>> results = fair_runner.run(species, efficacy_erf, species_quantities)
    """
    def __init__(self, start_year: int, end_year: int, background_species_quantities: dict = None):
        self.start_year = start_year
        self.end_year = end_year
        self.background_species_quantities = background_species_quantities
        self.species_list = None
        self.properties = None
        self.f = None

    def _setup_model(self):
        # --- Initialize FaIR instance ---
        f = self.f = FAIR()
        start_year = self.start_year
        end_year = self.end_year
        background_species_quantities = self.background_species_quantities

        # --- Define time horizon of the simulation ---
        f.define_time(start_year, end_year, 1)

        # --- Define scenario to be run ---
        f.define_scenarios(["central"])

        # --- Define configuration to be run ---
        f.define_configs(["central"])
        # f.define_configs(["high", "central", "low"])

        # --- Define species that will be included in the simulation ---
        species_list = self.species_list = [
            "CO2",  # Includes world and aviation emissions
            "World CH4",  # Includes background emissions only
            "Contrails",
            "NOx - ST O3 increase",
            "NOx - CH4 decrease and induced",
            "H2O",
            "Sulfur",
            "Soot",
            "Aerosols",
        ]
        properties = self.properties = {
            "CO2": {
                "type": "co2",
                "input_mode": "emissions",
                "greenhouse_gas": True,
                "aerosol_chemistry_from_emissions": False,
                "aerosol_chemistry_from_concentration": False,
            },
            "World CH4": {
                "type": "ch4",
                "input_mode": "emissions",
                "greenhouse_gas": True,
                "aerosol_chemistry_from_emissions": False,
                "aerosol_chemistry_from_concentration": True,
            },
            "Contrails": {
                "type": "contrails",
                "input_mode": "forcing",
                "greenhouse_gas": False,
                "aerosol_chemistry_from_emissions": False,
                "aerosol_chemistry_from_concentration": False,
            },
            "NOx - ST O3 increase": {
                "type": "ozone",
                "input_mode": "forcing",
                "greenhouse_gas": False,
                "aerosol_chemistry_from_emissions": False,
                "aerosol_chemistry_from_concentration": False,
            },
            "NOx - CH4 decrease and induced": {
                "type": "unspecified",
                "input_mode": "forcing",
                "greenhouse_gas": False,
                "aerosol_chemistry_from_emissions": False,
                "aerosol_chemistry_from_concentration": False,
            },
            "H2O": {
                "type": "h2o stratospheric",
                "input_mode": "forcing",
                "greenhouse_gas": False,
                "aerosol_chemistry_from_emissions": False,
                "aerosol_chemistry_from_concentration": False,
            },
            "Sulfur": {
                "type": "sulfur",
                "input_mode": "emissions",
                "greenhouse_gas": False,
                "aerosol_chemistry_from_emissions": True,
                "aerosol_chemistry_from_concentration": False,
            },
            "Soot": {
                "type": "black carbon",
                "input_mode": "emissions",
                "greenhouse_gas": False,
                "aerosol_chemistry_from_emissions": True,
                "aerosol_chemistry_from_concentration": False,
            },
            "Aerosols": {
                "type": "ari",
                "input_mode": "calculated",
                "greenhouse_gas": False,
                "aerosol_chemistry_from_emissions": False,
                "aerosol_chemistry_from_concentration": False,
            },
        }
        f.define_species(species_list, properties)

        # --- Modify run control options ---
        f.ghg_method = "leach2021"
        f.aci_method = "myhre1998"

        # --- Create input and output data arrays ---
        f.allocate()

        # --- Fill climate configs ---
        fill(f.climate_configs["ocean_heat_transfer"], [2, 2, 1.1], config="central")
        fill(f.climate_configs["ocean_heat_capacity"], [6, 11, 75], config="central")
        fill(f.climate_configs["deep_ocean_efficacy"], 0.8, config="central")

        # --- Fill default species configs ---
        # - CO2 -
        fill(
            f.species_configs["partition_fraction"],
            [0.2173, 0.2240, 0.2824, 0.2763],
            specie="CO2",
        )
        fill(
            f.species_configs["unperturbed_lifetime"],
            [1e9, 394.4, 36.54, 4.304],
            specie="CO2",
        )
        fill(f.species_configs["baseline_concentration"], 278.3, specie="CO2")
        fill(f.species_configs["forcing_reference_concentration"],278.3, specie="CO2")
        fill(f.species_configs["molecular_weight"], 44.009, specie="CO2")
        fill(f.species_configs["greenhouse_gas_radiative_efficiency"],1.37e-05, specie="CO2")
        f.calculate_iirf0()
        f.calculate_g()
        f.calculate_concentration_per_emission()
        fill(f.species_configs["iirf_0"], 29, specie="CO2")
        fill(f.species_configs["iirf_airborne"], [0.000819 / 2], specie="CO2")
        fill(f.species_configs["iirf_uptake"], [0.00846 / 2], specie="CO2")
        fill(f.species_configs["iirf_temperature"], [4 / 2], specie="CO2")
        fill(f.species_configs["aci_scale"], -3.14762148)

        # - CH4 -
        fill(f.species_configs["partition_fraction"], [1, 0, 0, 0], specie="World CH4")
        fill(f.species_configs["unperturbed_lifetime"], 8.25, specie="World CH4")
        fill(f.species_configs["baseline_concentration"], 729, specie="World CH4")
        fill(f.species_configs["forcing_reference_concentration"], 729, specie="World CH4")
        fill(f.species_configs["molecular_weight"], 16.043, specie="World CH4")
        fill(f.species_configs["greenhouse_gas_radiative_efficiency"],0.00038864402860869495, specie="World CH4")
        f.calculate_iirf0()
        f.calculate_g()
        f.calculate_concentration_per_emission()
        fill(f.species_configs["iirf_airborne"], 0.00032, specie="World CH4")
        fill(f.species_configs["iirf_uptake"], 0, specie="World CH4")
        fill(f.species_configs["iirf_temperature"], -0.3, specie="World CH4")
        fill(f.species_configs["erfari_radiative_efficiency"], -0.002653 / 1023.2219696044921, specie="World CH4")
        fill(f.species_configs["aci_scale"], -2.09841432)

        # - Sulfur -
        erf_aci_sulfur = 0.0
        fill(f.species_configs["erfari_radiative_efficiency"], -0.0199 + erf_aci_sulfur, specie="Sulfur")
        fill(f.species_configs["aci_shape"], 0.0, specie="Sulfur")

        # - Soot -
        erf_aci_BC = 0.0
        fill(f.species_configs["erfari_radiative_efficiency"], 0.1007 + erf_aci_BC, specie="Soot")
        fill(f.species_configs["aci_shape"], 0.0, specie="Soot")

        # --- Initialise all emissions and forcing to zero ---
        self.initialise_emissions_and_forcing()

        # Set background CH4 emissions (without aviation)
        fill(
            f.emissions,
            background_species_quantities["background_CH4"][1:],
            specie="World CH4",
            config=f.configs[0],
            scenario=f.scenarios[0],
        )

        # Set background CO2 emissions (without aviation)
        fill(
            f.emissions,
            background_species_quantities["background_CO2"][1:],
            specie="CO2",
            config=f.configs[0],
            scenario=f.scenarios[0],
        )

    def run(self,
            species: str = None,
            efficacy_erf: Union[int, float] = 1.0,
            emission_profile: Union[list, np.ndarray] = None) -> dict:
        """
        Run FaIR climate model previously configured, for a (single) given species and its emission profile.
        :param species: name of the species to be studied. If None, run background scenario with no additional species.
        :param efficacy_erf: efficacy of the species for effective radiative forcing (default: 1.0)
        :param emission_profile: array of annual emissions/forcing values for the species.
        :return: results: dict with 'effective_radiative_forcing' and 'temperature' (arrays of annual values)
        """
        # --- Setup model for fresh start ---
        self._setup_model()

        # --- Prepare inputs ---
        f = self.f
        species_list = self.species_list
        properties = self.properties
        if species not in species_list + [None]:  # None is allowed for run with only background species
            warnings.warn(f"Species '{species}' not recognized and won't have any effect. Available species: {species_list}")

        # --- Set efficacy erf for current species ---
        if species in species_list:
            fill(f.species_configs["forcing_efficacy"], efficacy_erf, specie=species)

        # --- Set emissions/forcing inputs for current species ---
        # - special case for CO2: adds to background CO2 -
        if species == "CO2":
            total_CO2 = f.emissions.loc[dict(specie="CO2", config=f.configs[0], scenario=f.scenarios[0])].data  # background CO2 emissions
            total_CO2 += emission_profile[1:]  # add aviation CO2 emissions
            fill(f.emissions, total_CO2, specie="CO2", config=f.configs[0], scenario=f.scenarios[0])

        # - Species not recognized -
        elif species not in species_list:
            pass  # species not recognized, do nothing

        # - Species using forcing as input instead of emissions -
        elif properties[species]["input_mode"] == "forcing":
            fill(
                f.forcing,
                emission_profile,
                specie=species,
                config=f.configs[0],
                scenario=f.scenarios[0],
            )

        # - Species using emissions as input -
        else:
            fill(
                f.emissions,
                emission_profile[1:],
                specie=species,
                config=f.configs[0],
                scenario=f.scenarios[0],
            )

        # --- Initialise state variables to zero ---
        initialise(f.forcing, 0)
        initialise(f.temperature, 0)
        initialise(f.cumulative_emissions, 0)
        initialise(f.airborne_emissions, 0)

        # --- Run model ---
        f.run(progress=False)

        # --- Results ---
        results = {
            "effective_radiative_forcing": f.forcing_sum.loc[dict(config=f.configs[0])].data,
            "temperature": f.temperature.loc[dict(config=f.configs[0], layer=0)].data,
        }

        return results

    def initialise_emissions_and_forcing(self):
        f = self.f
        for specie in self.species_list:
            if self.properties[specie]["input_mode"] == "forcing":
                fill(f.forcing, 0, specie=specie, config=f.configs[0], scenario=f.scenarios[0])
            else:
                fill(f.emissions, 0, specie=specie, config=f.configs[0], scenario=f.scenarios[0])


def background_species_quantities_function(start_year: int, end_year: int, rcp: str = None) -> dict:
    """
    Get background species quantities (CO2 and CH4) from RCP scenarios.
    :param start_year: start year of the simulation
    :param end_year: end year of the simulation
    :param rcp: Representative Concentration Pathway. Must be one of 'RCP26', 'RCP45', 'RCP60', 'RCP85', or 'None' for no background species.
    :return: background_species_quantities: dict of annual emission values for the background species (CO2, CH4), from start_year to end_year

    ---
    Example usage:
    >>> from aerometrics.climate_models.fair_climate_model import background_species_quantities_function
    >>> background_species_quantities = background_species_quantities_function(2020, 2050, 'RCP45')
    """

    # --- Validate inputs ---
    if start_year < RCP_START_YEAR:
        raise ValueError(f"start_year must be >= {RCP_START_YEAR}")

    # --- Initialize variables ---
    background_species_quantities = {
        "background_CO2": np.zeros(end_year - start_year + 1),
        "background_CH4": np.zeros(end_year - start_year + 1)
    }
    rcp_data_path = None

    # --- Read data ---
    if rcp == "RCP26":
        rcp_data_path = pth.join(RCP.__path__[0], "RCP26.csv")
    elif rcp == "RCP45":
        rcp_data_path = pth.join(RCP.__path__[0], "RCP45.csv")
    elif rcp == "RCP60":
        rcp_data_path = pth.join(RCP.__path__[0], "RCP60.csv")
    elif rcp == "RCP85":
        rcp_data_path = pth.join(RCP.__path__[0], "RCP85.csv")
    else:
        warnings.warn("RCP scenario not recognized (available: RCP26, RCP45, RCP60, RCP85). "
                      "Background species will be set to zero.")

    if rcp_data_path:
        rcp_data_df = pd.read_csv(rcp_data_path)

        # World CO2
        background_species_quantities["background_CO2"] = (
                rcp_data_df["FossilCO2"][start_year - RCP_START_YEAR : end_year - RCP_START_YEAR + 1].values
                + rcp_data_df["OtherCO2"][start_year - RCP_START_YEAR : end_year - RCP_START_YEAR + 1].values
            ) * 44 / 12  # Conversion from GtC to GtCO2

        # World CH4
        background_species_quantities["background_CH4"] = rcp_data_df["CH4"][
                                           start_year - RCP_START_YEAR: end_year - RCP_START_YEAR + 1].values  # Unit: MtCH4

    return background_species_quantities


class FairClimateModel(ClimateModel):
    """
    Climate model using FaIR to compute the RF, ERF and temperature increase for a given species and its emission profile.
    https://docs.fairmodel.net/en/latest/

    ---
    Example usage:
    >>> import numpy as np
    >>> from aerometrics.climate_models.fair_climate_model import FairClimateModel
    >>> from aerometrics.climate_models.fair_climate_model import background_species_quantities_function
    >>> start_year = 2020
    >>> end_year = 2050
    >>> species = "CO2"
    >>> species_quantities = np.array([1e12] * (end_year - start_year + 1))  # 1 GtCO2 per year
    >>> species_settings = {
    ...     "sensitivity_rf": 1.0,
    ...     "ratio_erf_rf": 1.0,
    ...     "efficacy_erf": 1.0,
    ... }
    >>> background_species_quantities = background_species_quantities_function(start_year, end_year, 'RCP45')
    >>> model_settings = {
    ...     "background_species_quantities": background_species_quantities,
    ... }
    >>> fair_climate_model = FairClimateModel(start_year, end_year, species, species_settings, model_settings)
    >>> results = fair_climate_model.run()
    """

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
        "CO2": {"ratio_erf_rf": float},
        "Contrails": {"sensitivity_rf": float, "ratio_erf_rf": float, "efficacy_erf": float},
        "NOx - ST O3 increase": {"sensitivity_rf": float, "ratio_erf_rf": float, "efficacy_erf": float},
        "NOx - CH4 decrease and induced": {"ch4_loss_per_nox": float, "ratio_erf_rf": float, "efficacy_erf": float},
        "Soot": {"ratio_erf_rf": float, "efficacy_erf": float},
        "Sulfur": {"ratio_erf_rf": float, "efficacy_erf": float},
        "H2O": {"sensitivity_rf": float, "ratio_erf_rf": float, "efficacy_erf": float},
    }
    mandatory_model_settings = {"rcp": Union[str, None]}
    optional_model_settings = {
        "background_species_quantities": dict,  # overrode by rcp if rcp is provided
        "background_effective_radiative_forcing": Union[list, np.ndarray],
        "background_temperature": Union[list, np.ndarray]
    }

    def __init__(
        self,
        start_year: int,
        end_year: int,
        species: str,
        emission_profile: Union[list, np.ndarray],
        species_settings: dict,
        model_settings: dict,
    ):
        # --- Check if RCP is provided, else background species quantities must be provided ---
        if "rcp" not in model_settings.keys():
            self.mandatory_model_settings = {"background_species_quantities": dict}
        super().__init__(start_year, end_year, species, emission_profile, species_settings, model_settings)

    def run(self, return_df: bool = False) -> dict | pd.DataFrame:
        """
        Compute the RF, ERF and temperature increase for a given species and its quantities using the FaIR climate model.

        Parameters
        ----------
        return_df : bool, optional
            If True, returns the results as a pandas DataFrame with years as index. Default is False (returns a dict).

        Returns
        -------
        output_data : dict
            Dictionary containing the results of the LWE climate model.
        """

        # --- Extract model settings ---
        model_settings = self.model_settings
        rcp = model_settings.get("rcp", None)
        if "rcp" in model_settings.keys() and "background_species_quantities" in model_settings.keys():
            warnings.warn(f"RCP scenario '{rcp}' provided. Background species quantities will be set accordingly, overriding any provided background_species_quantities in model_settings.")
            background_species_quantities = background_species_quantities_function(
                self.start_year, self.end_year, rcp
            )
        elif "rcp" in model_settings.keys():
            background_species_quantities = background_species_quantities_function(
                self.start_year, self.end_year, rcp
            )
        else:
            background_species_quantities = model_settings["background_species_quantities"]

        # --- Extract species settings ---
        species_settings = self.species_settings
        sensitivity_rf = species_settings.get("sensitivity_rf", None)  # replace 2nd argument with default if needed
        ratio_erf_rf = species_settings.get("ratio_erf_rf", 1.0)
        efficacy_erf = species_settings.get("efficacy_erf", 1.0)
        ch4_loss_per_nox = species_settings.get("ch4_loss_per_nox", None)  # only for NOx - CH4 decrease and induced

        # --- Extract simulation settings ---
        start_year = self.start_year
        end_year = self.end_year
        species = self.species
        emission_profile = self.emission_profile
        years = list(range(start_year, end_year + 1))

        # --- Prepare inputs depending on species ---
        processed_emission_profile = None

        if species == "CO2":
            processed_emission_profile = (
                    emission_profile / 10 ** 12
            )  # Conversion from kgCO2 to GtCO2
        elif species == "Soot":
            processed_emission_profile = (
                    emission_profile / 10 ** 9
            )  # Conversion from kgSO2 to MtSO2
        elif species == "Sulfur":
            processed_emission_profile = (
                    emission_profile / 10 ** 9
            )  # Conversion from kgBC to MtBC
        elif species == "Contrails":
            rf = sensitivity_rf * emission_profile
            erf = rf * ratio_erf_rf
            processed_emission_profile = erf  # W/m2
        elif species == "H2O":
            rf = sensitivity_rf * emission_profile
            erf = rf * ratio_erf_rf
            processed_emission_profile = erf  # W/m2
        elif species == "NOx - ST O3 increase":
            rf = sensitivity_rf * emission_profile
            erf = rf * ratio_erf_rf
            processed_emission_profile = erf  # W/m2
        elif species == "NOx - CH4 decrease and induced":
            min_year = min(start_year, 1939)
            max_year = max(end_year, 2051)
            tau_reference_year = [min_year, 1940, 1980, 1994, 2004, 2050, max_year]
            tau_reference_values = [11, 11, 10.1, 10, 9.85, 10.25, 10.25]
            tau_function = interp1d(tau_reference_year, tau_reference_values, kind="linear")
            tau = tau_function(years)
            ch4_molar_mass = 16.04e-3  # [kg/mol]
            air_molar_mass = 28.97e-3  # [kg/mol]
            atmosphere_total_mass = 5.1352e18  # [kg]
            radiative_efficiency = 3.454545e-4  # radiative efficiency [W/m^2/ppb] with AR6 value (5.7e-4) without indirect effects
            A_CH4_unit = (
                    radiative_efficiency
                    * 1e9
                    * air_molar_mass
                    / (ch4_molar_mass * atmosphere_total_mass)
            )  # RF per unit mass increase in atmospheric abundance of CH4 [W/m^2/kg]
            A_CH4 = A_CH4_unit * ch4_loss_per_nox * emission_profile
            f1 = 0.5  # Indirect effect on ozone
            f2 = 0.15  # Indirect effect on stratospheric water
            radiative_forcing_from_year = np.zeros(
                (len(emission_profile), len(emission_profile))
            )
            # Radiative forcing induced in year j by the species emitted in year i
            for i in range(0, len(emission_profile)):
                for j in range(0, len(emission_profile)):
                    if i <= j:
                        radiative_forcing_from_year[i, j] = (
                                (1 + f1 + f2) * A_CH4[i] * np.exp(-(j - i) / tau[j])
                        )
            radiative_forcing = np.zeros(len(emission_profile))
            for k in range(0, len(emission_profile)):
                radiative_forcing[k] = np.sum(
                    radiative_forcing_from_year[:, k]
                )
            effective_radiative_forcing = radiative_forcing * ratio_erf_rf
            processed_emission_profile = effective_radiative_forcing  # W/m2

        # --- Run FaIR model ---
        fair_runner = FairRunner(start_year, end_year, background_species_quantities)
        results = fair_runner.run(species, efficacy_erf, processed_emission_profile)
        temperature_with_species = results["temperature"]
        effective_radiative_forcing_with_species = results["effective_radiative_forcing"]

        # --- Counterfactual scenario (without the species) ---
        # If background ERF and temperature are provided in model_settings, use them
        if {"background_effective_radiative_forcing", "background_temperature"} <= model_settings.keys():
            temperature_without_species = model_settings["background_temperature"]
            effective_radiative_forcing_without_species = model_settings["background_effective_radiative_forcing"]
        # Else, run FaIR with no additional species
        else:
            results_background = fair_runner.run()  # Run with no additional species
            temperature_without_species = results_background["temperature"]
            effective_radiative_forcing_without_species = results_background["effective_radiative_forcing"]

        # --- Compute RF, ERF and temperature increase due to the species ---
        temperature = temperature_with_species - temperature_without_species

        # For some species, the ERF is directly obtained from the inputs
        if species in [
            "Contrails",
            "NOx - ST O3 increase",
            "NOx - CH4 decrease and induced",
            "H2O",
        ]:
            effective_radiative_forcing = processed_emission_profile.reshape(-1, 1)
        # For other species, the ERF is the difference between the FaIR runs with and without the species
        else:
            effective_radiative_forcing = (
                    effective_radiative_forcing_with_species
                    - effective_radiative_forcing_without_species
            )

        radiative_forcing = effective_radiative_forcing / ratio_erf_rf

        # --- Return results ---
        output_data = {
            "radiative_forcing": radiative_forcing.flatten(),
            "effective_radiative_forcing": effective_radiative_forcing.flatten(),
            "temperature": temperature.flatten(),
        }
        if return_df:
            output_data = pd.DataFrame(output_data, index=years)
            output_data.index.name = 'Year'

        return output_data


def background_fair_climate_model(
    start_year: int, end_year: int, background_species_quantities: dict
) -> dict:
    """
    Compute the background ERF and temperature increase using the FaIR climate model with no additional species.
    :param start_year: start year of the simulation
    :param end_year: end year of the simulation
    :param background_species_quantities: dictionary with annual emission values for the background species (CO2, CH4), from start_year to end_year
    :return: background_climate: dict with 'background_effective_radiative_forcing' and 'background_temperature' (arrays of annual values)
    """
    # --- Initialize results dict ---
    background_climate = {
        "background_effective_radiative_forcing": np.zeros(end_year - start_year + 1),
        "background_temperature": np.zeros(end_year - start_year + 1),
    }

    # --- Run FaIR with no additional species ---
    fair_runner = FairRunner(start_year, end_year, background_species_quantities)  # Initialize FaIR runner with model settings
    results = fair_runner.run()  # Run with no additional species
    background_climate["background_effective_radiative_forcing"] = results["effective_radiative_forcing"].flatten()
    background_climate["background_temperature"] = results["temperature"].flatten()

    return background_climate