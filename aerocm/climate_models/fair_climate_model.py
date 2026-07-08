"""
This module contains the FairClimateModel class, which implements a climate model using the FaIR (Finite Amplitude Impulse Response) model.
"""

import warnings
import os.path as pth
import numpy as np
import pandas as pd
from fair import FAIR
from fair.interface import fill, initialise
from scipy.interpolate import interp1d
from aerocm.utils.classes import ClimateModel
from aerocm.climate_data import background_scenarios
from aerocm.climate_data import concentration


BACKGROUND_SCENARIO_START_YEAR = 1750
BACKGROUND_SCENARIO_END_YEAR = 2500

class FairClimateModel(ClimateModel):
    """
    Climate model using FaIR to compute the RF, ERF and temperature increase for a given species and its emission
    profile, accounting for the background scenario.

    Notes
    -----
    References:
        - Leach et al. (2021). https://doi.org/10.5194/gmd-14-3007-2021
        - Model implementation https://(docs).fairmodel.net/en/latest/
    """

    # --- Default parameters ---
    available_species = [
        "CO2",
        "Contrails",
        "NOx - ST O3",
        "NOx - CH4 and induced",
        "H2O",
        "Soot - ARI",
        "Soot - ACI",
        "Sulfur - ARI",
        "Sulfur - ACI",
        "H2 leakage - ST O3",
        "H2 leakage - CH4 and induced",
    ]
    available_species_settings = {
        "CO2": {"ratio_erf_rf": {"type": float, "default": 1.0}},
        "Contrails": {"sensitivity_rf": {"type": float, "default": 2.23e-12},
                      "ratio_erf_rf": {"type": float, "default": 0.42},
                      "efficacy_erf": {"type": float, "default": 1.0}},
        "NOx - ST O3": {"sensitivity_rf": {"type": float, "default": 7.64e-12},
                                 "ratio_erf_rf": {"type": float, "default": 1.37},
                                 "efficacy_erf": {"type": float, "default": 1.0}},
        "NOx - CH4 and induced": {"ch4_production_per_nox": {"type": float, "default": -3.9},
                                           "ratio_erf_rf": {"type": float, "default": 1.18},
                                           "efficacy_erf": {"type": float, "default": 1.0}},
        "H2O": {"sensitivity_rf": {"type": float, "default": 5.2e-15}, "ratio_erf_rf": {"type": float, "default": 1.0},
                "efficacy_erf": {"type": float, "default": 1.0}},
        "Soot - ARI": {"sensitivity_rf": {"type": float, "default": 1.0e-10},
                       "ratio_erf_rf": {"type": float, "default": 1.0},
                       "efficacy_erf": {"type": float, "default": 1.0}},
        "Soot - ACI": {"sensitivity_rf": {"type": float, "default": 0.0},
                       "ratio_erf_rf": {"type": float, "default": 1.0},
                       "efficacy_erf": {"type": float, "default": 1.0}},
        "Sulfur - ARI": {"sensitivity_rf": {"type": float, "default": -2.0e-11},
                   "ratio_erf_rf": {"type": float, "default": 1.0}, "efficacy_erf": {"type": float, "default": 1.0}},
        "Sulfur - ACI": {"sensitivity_rf": {"type": float, "default": 0.0},
                   "ratio_erf_rf": {"type": float, "default": 1.0}, "efficacy_erf": {"type": float, "default": 1.0}},
        "H2 leakage - ST O3": {"sensitivity_rf": {"type": float, "default": 0.4e-12},
                                 "ratio_erf_rf": {"type": float, "default": 1.37},
                                 "efficacy_erf": {"type": float, "default": 1.0}},
        "H2 leakage - CH4 and induced": {"ch4_production_per_nox": {"type": float, "default": 0.34},
                                           "ratio_erf_rf": {"type": float, "default": 1.18},
                                           "efficacy_erf": {"type": float, "default": 1.0}}
    }
    available_model_settings = {
        "contrails_saturation_factor": {"type": float, "default": 1.0},
        "background_nox_correction_factor": {"type": float, "default": 0.0},
        "background_scenario": {"type": (str, type(None)), "default": "RCP45"},
        # overrode by background_species_quantities if background_species_quantities is provided
        "background_species_quantities": {"type": dict},
        "background_effective_radiative_forcing": {"type": (list, np.ndarray)},
        "background_temperature": {"type": (list, np.ndarray)}
    }

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
            Dictionary containing the results of the FaIR climate model.
        """

        # --- Extract species settings ---
        specie_settings = self.specie_settings
        sensitivity_rf = specie_settings.get("sensitivity_rf", 0.0)  # replace 2nd argument with default if needed
        ratio_erf_rf = specie_settings.get("ratio_erf_rf", 1.0)
        efficacy_erf = specie_settings.get("efficacy_erf", 1.0)
        ch4_production_per_nox = specie_settings.get("ch4_production_per_nox", 0.0)  # only for NOx/H2 leakage - CH4 and induced

        # --- Extract simulation settings ---
        start_year = self.start_year
        end_year = self.end_year
        specie_name = self.specie_name
        specie_inventory = self.specie_inventory
        years = list(range(start_year, end_year + 1))

        # --- Extract model settings ---
        model_settings = self.model_settings
        contrails_saturation_factor = model_settings.get("contrails_saturation_factor", 1.0)
        background_nox_correction_factor = model_settings.get("background_nox_correction_factor", 0.0)
        background_species_quantities = self.get_background_species_quantities(
            model_settings,
            start_year,
            end_year
        )

        # --- Prepare inputs depending on species ---
        processed_inventory = None

        if specie_name == "CO2":
            processed_inventory = (
                    specie_inventory / 10 ** 12
            )  # Conversion from kgCO2 to GtCO2

        elif specie_name == "Soot - ARI":
            processed_inventory = (
                    specie_inventory / 10 ** 9
            )  # Conversion from kgSO2 to MtSO2

        elif specie_name == "Sulfur - ARI":
            processed_inventory = (
                    specie_inventory / 10 ** 9
            )  # Conversion from kgBC to MtBC

        elif specie_name == "Contrails":
            contrails_saturation_reference_year = 2018
            years_array = np.array(years)
            idx_ref_contrails = np.where(years_array == contrails_saturation_reference_year)[0][0]
            inventory_ref = specie_inventory[idx_ref_contrails]
            if inventory_ref == 0: #in case of pulse emissions etc. emissions/km may be zero
                saturation_inventory = specie_inventory
                inventory_ref = 1.0
            else: 
                saturation_inventory = (specie_inventory / inventory_ref) ** contrails_saturation_factor
            rf = sensitivity_rf * inventory_ref * saturation_inventory
            erf = rf * ratio_erf_rf
            processed_inventory = erf  # W/m2

        elif specie_name == "H2O" or specie_name == "Soot - ACI" or specie_name == "Sulfur - ACI" or specie_name == "H2 leakage - ST O3":
            rf = sensitivity_rf * specie_inventory
            erf = rf * ratio_erf_rf
            processed_inventory = erf  # W/m2

        elif specie_name == "H2 leakage - CH4 and induced":
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
            A_CH4 = A_CH4_unit * ch4_production_per_nox * specie_inventory
            f1 = 0.5  # Indirect effect on ozone
            f2 = 0.15  # Indirect effect on stratospheric water
            radiative_forcing_from_year = np.zeros(
                (len(specie_inventory), len(specie_inventory))
            )
            # Radiative forcing induced in year j by the species emitted in year i
            for i in range(0, len(specie_inventory)):
                for j in range(0, len(specie_inventory)):
                    if i <= j:
                        radiative_forcing_from_year[i, j] = (
                                (1 + f1 + f2) * A_CH4[i] * np.exp(-(j - i) / tau[j])
                        )
            radiative_forcing = np.zeros(len(specie_inventory))
            for k in range(0, len(specie_inventory)):
                radiative_forcing[k] = np.sum(
                    radiative_forcing_from_year[:, k]
                )
            effective_radiative_forcing = radiative_forcing * ratio_erf_rf
            processed_inventory = effective_radiative_forcing  # W/m2

        else:
            nox_background_reference_year = 2018
            nox_background = background_species_quantities["background_NOx"]
            dt_land = self.get_dt_land(nox_background, years, nox_background_reference_year)
            nox_correction = (dt_land * background_nox_correction_factor + 1)

            if specie_name == "NOx - ST O3":
                rf = sensitivity_rf * specie_inventory
                erf = rf * ratio_erf_rf
                processed_inventory =  erf * nox_correction # W/m2

            elif specie_name == "NOx - CH4 and induced":
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
                A_CH4 = A_CH4_unit * ch4_production_per_nox * specie_inventory
                f1 = 0.5  # Indirect effect on ozone
                f2 = 0.15  # Indirect effect on stratospheric water
                radiative_forcing_from_year = np.zeros(
                    (len(specie_inventory), len(specie_inventory))
                )
                # Radiative forcing induced in year j by the species emitted in year i
                for i in range(0, len(specie_inventory)):
                    for j in range(0, len(specie_inventory)):
                        if i <= j:
                            radiative_forcing_from_year[i, j] = (
                                    (1 + f1 + f2) * A_CH4[i] * np.exp(-(j - i) / tau[j])
                            )
                radiative_forcing = np.zeros(len(specie_inventory))
                for k in range(0, len(specie_inventory)):
                    radiative_forcing[k] = np.sum(
                        radiative_forcing_from_year[:, k]
                    )
                effective_radiative_forcing = radiative_forcing * ratio_erf_rf
                processed_inventory = effective_radiative_forcing * nox_correction # W/m2

        # --- Run FaIR model ---
        fair_runner = FairRunner(start_year, end_year, background_species_quantities)
        results = fair_runner.run(specie_name, sensitivity_rf, ratio_erf_rf, efficacy_erf, processed_inventory)
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
        if specie_name in [
            "Contrails",
            "NOx - ST O3",
            "NOx - CH4 and induced",
            "H2O",
            "Soot - ACI",
            "Sulfur - ACI"
            "H2 leakage - ST O3",
            "H2 leakage - CH4 and induced",
        ]:
            effective_radiative_forcing = processed_inventory.reshape(-1, 1)
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

    @staticmethod
    def get_dt_land(inventory, years, reference_year):
        """
        Computes a ratio of emissions growth versus a reference year, used to parametrse
        
        """
        years_array = np.array(years)
        idx_ref = np.where (years_array == reference_year)[0][0]
        emission_ref = inventory[idx_ref]

        dt_land = (inventory - emission_ref) / emission_ref
        dt_land = np.nan_to_num(dt_land, 0.0) #remove NaNs (if no or null background emissions)
        return dt_land

    @staticmethod
    def get_background_species_quantities(model_settings: dict = None, start_year: int = None, end_year: int = None) -> dict:
        """
        Get the background species quantities from the model settings or from the background scenario.

        Parameters
        ----------
        model_settings : dict
            Dictionary containing model settings.
        start_year : int
            Start year of the simulation.
        end_year : int
            End year of the simulation.

        Returns
        -------
        background_species_quantities : dict
            Dictionary containing the background species quantities (CO2 and CH4) for each year of the simulation.

        """
        scenario = model_settings.get("background_scenario")
        
        if "background_species_quantities" in model_settings:
            if scenario:
                warnings.warn(
                    f"Both scenario and background species provided in model_settings. "
                    f"The background species provided will override scenario '{scenario}'.")
            
            background_species_quantities = model_settings["background_species_quantities"]
            
        elif scenario:
            background_species_quantities = background_species_quantities_function(
                start_year,
                end_year,
                scenario
            )
            
        else:
            raise ValueError("Either 'background_scenario' or 'background_species_quantities' must be provided in model_settings.")

        return background_species_quantities


class FairRunner:
    """
    Class to run the FaIR climate model for a (single) given species and its emission profile.

    Parameters
    ----------
    start_year : int
        Start year of the simulation.
    end_year : int
        End year of the simulation.
    background_species_quantities : dict, optional
        Dictionary containing the background species quantities (CO2 and CH4) for each year of the simulation.

    Attributes
    ----------
    start_year : int
        Start year of the simulation.
    end_year : int
        End year of the simulation.
    background_species_quantities : dict
        Dictionary containing the background species quantities (CO2 and CH4) for each year of the simulation.
    species_list : list
        List of species included in the simulation.
    properties : dict
        Dictionary containing the properties of each species.
    f : FAIR
        Instance of the FAIR model.

    Notes
    -----
    This class is used internally by the FairClimateModel class, and is not intended to be used directly.
    """
    def __init__(self, start_year: int, end_year: int, background_species_quantities: dict = None):
        self.start_year = start_year
        self.end_year = end_year
        self.background_species_quantities = background_species_quantities
        self.species_list = None
        self.properties = None
        self.f = None

    def _setup_model(self):
        """
        Setup and configure the FaIR climate model instance.
        """
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
            "NOx - ST O3",
            "NOx - CH4 and induced",
            "H2O",
            "Soot - ARI",
            "Soot - ACI",
            "Sulfur - ARI",
            "Sulfur - ACI",
            "H2 leakage - ST O3",
            "H2 leakage - CH4 and induced",
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
            "NOx - ST O3": {
                "type": "ozone",
                "input_mode": "forcing",
                "greenhouse_gas": False,
                "aerosol_chemistry_from_emissions": False,
                "aerosol_chemistry_from_concentration": False,
            },
            "NOx - CH4 and induced": {
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
            "Soot - ARI": {
                "type": "black carbon",
                "input_mode": "emissions",
                "greenhouse_gas": False,
                "aerosol_chemistry_from_emissions": True,
                "aerosol_chemistry_from_concentration": False,
            },
            "Soot - ACI": {
                "type": "unspecified",
                "input_mode": "forcing",
                "greenhouse_gas": False,
                "aerosol_chemistry_from_emissions": False,
                "aerosol_chemistry_from_concentration": False,
            },
            "Sulfur - ARI": {
                "type": "sulfur",
                "input_mode": "emissions",
                "greenhouse_gas": False,
                "aerosol_chemistry_from_emissions": True,
                "aerosol_chemistry_from_concentration": False,
            },
            "Sulfur - ACI": {
                "type": "unspecified",
                "input_mode": "forcing",
                "greenhouse_gas": False,
                "aerosol_chemistry_from_emissions": False,
                "aerosol_chemistry_from_concentration": False,
            },
            "H2 leakage - ST O3": {
                "type": "unspecified",
                "input_mode": "forcing",
                "greenhouse_gas": False,
                "aerosol_chemistry_from_emissions": False,
                "aerosol_chemistry_from_concentration": False,
            },
            "H2 leakage - CH4 and induced": {
                "type": "unspecified",
                "input_mode": "forcing",
                "greenhouse_gas": False,
                "aerosol_chemistry_from_emissions": False,
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
        fill(f.climate_configs["ocean_heat_transfer"], [1.3, 1.6, 0.6], config="central")
        fill(f.climate_configs["ocean_heat_capacity"], [8, 14, 100], config="central")
        fill(f.climate_configs["deep_ocean_efficacy"], 1.1, config="central")

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

        # Update concentration data depending on the start year of the simulation
        # Use of EEA data (5-year linear interpolation between 1765 and 1975)
        concentration_data_path = pth.join(concentration.__path__[0], "concentration_data.csv")
        df = pd.read_csv(concentration_data_path, sep=";")
        df["Year"] = df["Year"].astype(int)
        row = df.loc[df["Year"] == start_year]
        if not row.empty:
            co2_concentration = row["CO2"].values[0] # 278.3 ppm per default in FaIR
            ch4_concentration = row["CH4"].values[0] # 729 ppb per default in FaIR
        else:
            raise ValueError(
                f"{self.start_year} is not a start year usable with the climate model parametrisation. "
                f"Choose a start year between 1765 and 2017."
            )

        fill(f.species_configs["baseline_concentration"], co2_concentration, specie="CO2")
        fill(f.species_configs["forcing_reference_concentration"],co2_concentration, specie="CO2")
        fill(f.species_configs["molecular_weight"], 44.009, specie="CO2")
        fill(f.species_configs["greenhouse_gas_radiative_efficiency"],1.37e-05, specie="CO2")
        f.calculate_iirf0()
        f.calculate_g()
        f.calculate_concentration_per_emission()
        fill(f.species_configs["iirf_0"], 29, specie="CO2")
        fill(f.species_configs["iirf_airborne"], [0.000819], specie="CO2")
        fill(f.species_configs["iirf_uptake"], [0.00846], specie="CO2")
        fill(f.species_configs["iirf_temperature"], [4], specie="CO2")
        fill(f.species_configs["aci_scale"], -2.09841432)

        # - CH4 -
        fill(f.species_configs["partition_fraction"], [1, 0, 0, 0], specie="World CH4")
        fill(f.species_configs["unperturbed_lifetime"], 8.25, specie="World CH4")
        fill(f.species_configs["baseline_concentration"], ch4_concentration, specie="World CH4")
        fill(f.species_configs["forcing_reference_concentration"], ch4_concentration, specie="World CH4")
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

        # - Sulfur and soot ARI -
        # Directly in the run method as it depends on the sensitivity_rf parameter

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

        # Set background NOx emissions (without aviation)
        #fill(
        #    f.emissions,
        #    background_species_quantities["background_NOx"][1:],
        #    specie="NOx",
        #    config=f.configs[0],
        #    scenario=f.scenarios[0],
        #)

    def run(self,
            specie_name: str = None,
            sensitivity_rf: int | float = 0.0,
            ratio_erf_rf: int | float = 1.0,
            efficacy_erf: int | float = 1.0,
            specie_inventory: list | np.ndarray = None) -> dict:
        """
        Run FaIR climate model previously configured, for a (single) given species and its emission profile.

        Parameters
        ----------
        specie_name: str, optional
            Name of the species to be studied. If None, run background scenario with no additional species.
        efficacy_erf: int | float, optional
            Efficacy of the species for effective radiative forcing (default: 1.0)
        specie_inventory: list | np.ndarray, optional
            Array of annual emissions/forcing values for the species.

        Returns
        -------
        results : dict
            Dictionary containing the results of the FaIR climate model run for the effective radiative forcing
            and temperature.
        """
        # --- Setup model for fresh start ---
        self._setup_model()

        # --- Prepare inputs ---
        f = self.f
        species_list = self.species_list
        properties = self.properties
        if specie_name not in species_list + [None]:  # None is allowed for run with only background species
            warnings.warn(f"Species '{specie_name}' not recognized and won't have any effect. Available species: {species_list}")

        if specie_name == "Soot - ARI":
            erf_ari_soot = sensitivity_rf * ratio_erf_rf * 10**9 # W/m² per MtSO2/yr, conversion from W/m² per kgSO2/yr
            fill(f.species_configs["erfari_radiative_efficiency"], erf_ari_soot, specie="Soot - ARI")
            fill(f.species_configs["aci_shape"], 0.0, specie="Soot - ARI")

        if specie_name == "Sulfur - ARI":
            erf_ari_sulfur = sensitivity_rf * ratio_erf_rf * 10**9 # W/m² per MtSO2/yr, conversion from W/m² per kgSO2/yr
            fill(f.species_configs["erfari_radiative_efficiency"], erf_ari_sulfur, specie="Sulfur - ARI")
            fill(f.species_configs["aci_shape"], 0.0, specie="Sulfur - ARI")

        # --- Set efficacy erf for current species ---
        if specie_name in species_list:
            fill(f.species_configs["forcing_efficacy"], efficacy_erf, specie=specie_name)

        # --- Set emissions/forcing inputs for current species ---
        # - special case for CO2: adds to background CO2 -
        if specie_name == "CO2":
            total_CO2 = f.emissions.loc[dict(specie="CO2", config=f.configs[0], scenario=f.scenarios[0])].data  # background CO2 emissions
            total_CO2 += specie_inventory[1:]  # add aviation CO2 emissions
            fill(f.emissions, total_CO2, specie="CO2", config=f.configs[0], scenario=f.scenarios[0])

        # - Species not recognized -
        elif specie_name not in species_list:
            pass  # species not recognized, do nothing

        # - Species using forcing as input instead of emissions -
        elif properties[specie_name]["input_mode"] == "forcing":
            fill(
                f.forcing,
                specie_inventory,
                specie=specie_name,
                config=f.configs[0],
                scenario=f.scenarios[0],
            )

        # - Species using emissions as input -
        else:
            fill(
                f.emissions,
                specie_inventory[1:],
                specie=specie_name,
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
        """
        Initialise all emissions and forcing to zero for all species.
        """
        f = self.f
        for specie in self.species_list:
            if self.properties[specie]["input_mode"] == "forcing":
                fill(f.forcing, 0, specie=specie, config=f.configs[0], scenario=f.scenarios[0])
            else:
                fill(f.emissions, 0, specie=specie, config=f.configs[0], scenario=f.scenarios[0])


def background_species_quantities_function(start_year: int, end_year: int, scenario: str = None) -> dict:
    """
    Get background species quantities (CO2 and CH4) from RCP or SSP scenarios.

    Parameters
    ----------
    start_year : int
        Start year of the simulation.
    end_year : int
        End year of the simulation.
        Background scenario to be used ('RCP26', 'RCP45', 'RCP60', 'RCP85', 'SSP119', 'SSP126', 'SSP245', 'SSP370', 'SSP434', 'SSP460', 'SSP534-over', 'SSP585'). Select None to set background species to zero.

    Returns
    -------
    background_species_quantities : dict
        Dictionary containing the background species quantities (CO2, CH4, NOx) for each year of the simulation.

    Example
    -------
    ```python
    >>> from aerocm.climate_models.fair_climate_model import background_species_quantities_function
    >>> background_species_quantities = background_species_quantities_function(2020, 2050, 'RCP45')
    ```
    """

    # --- Validate inputs ---
    if start_year < BACKGROUND_SCENARIO_START_YEAR:
        raise ValueError(f"start_year must be >= {BACKGROUND_SCENARIO_START_YEAR}")

    # --- Initialise variables ---
    background_species_quantities = {
        "background_CO2": np.zeros(end_year - start_year + 1),
        "background_CH4": np.zeros(end_year - start_year + 1),
        "background_NOx": np.zeros(end_year - start_year + 1)
    }

    background_scenario_data_path = None

    # --- Read data ---
    if scenario == "SSP119":
        background_scenario_data_path = pth.join(background_scenarios.__path__[0], "SSP119.csv")
    elif scenario == "SSP126":
        background_scenario_data_path = pth.join(background_scenarios.__path__[0], "SSP126.csv")
    elif scenario == "SSP245":
        background_scenario_data_path = pth.join(background_scenarios.__path__[0], "SSP245.csv")
    elif scenario == "SSP370":
        background_scenario_data_path = pth.join(background_scenarios.__path__[0], "SSP370.csv")
    elif scenario == "SSP434":
        background_scenario_data_path = pth.join(background_scenarios.__path__[0], "SSP434.csv")
    elif scenario == "SSP460":
        background_scenario_data_path = pth.join(background_scenarios.__path__[0], "SSP460.csv")
    elif scenario == "SSP534-over":
        background_scenario_data_path = pth.join(background_scenarios.__path__[0], "SSP534-over.csv")
    elif scenario == "SSP585":
        background_scenario_data_path = pth.join(background_scenarios.__path__[0], "SSP585.csv")
    elif scenario == "RCP26":
        background_scenario_data_path = pth.join(background_scenarios.__path__[0], "RCP26.csv")
    elif scenario == "RCP45":
        background_scenario_data_path = pth.join(background_scenarios.__path__[0], "RCP45.csv")
    elif scenario == "RCP60":
        background_scenario_data_path = pth.join(background_scenarios.__path__[0], "RCP60.csv")
    elif scenario == "RCP85":
        background_scenario_data_path = pth.join(background_scenarios.__path__[0], "RCP85.csv")
    else:
        warnings.warn("Scenario not recognised (available: SSP119, SSP126, SSP245, SSP370, SSP434, SSP460, SSP534-over, SSP585, RCP26, RCP45, RCP60, RCP85)")

    background_scenario_data_df = pd.read_csv(background_scenario_data_path)

    # World CO2
    background_species_quantities["background_CO2"] = (
            (background_scenario_data_df["CO2"][start_year - BACKGROUND_SCENARIO_START_YEAR : end_year - BACKGROUND_SCENARIO_START_YEAR + 1].values)/1000
        ) # Unit: GtCO2

    # World CH4
    background_species_quantities["background_CH4"] = background_scenario_data_df["CH4"][
                                       start_year - BACKGROUND_SCENARIO_START_YEAR: end_year - BACKGROUND_SCENARIO_START_YEAR + 1].values  # Unit: MtCH4

    # Background NOx
    background_species_quantities["background_NOx"] = background_scenario_data_df["NOx"][
                                       start_year - BACKGROUND_SCENARIO_START_YEAR: end_year - BACKGROUND_SCENARIO_START_YEAR + 1].values # Unit: MtNOx

    if end_year > BACKGROUND_SCENARIO_END_YEAR:
        # World CO2
        constant_co2 = (background_scenario_data_df["CO2"].values[-1] ) * np.ones(
            end_year - BACKGROUND_SCENARIO_END_YEAR)
        background_species_quantities["background_CO2"] = np.concatenate((background_species_quantities["background_CO2"],
                                                                    constant_co2))

        # World CH4
        constant_ch4 = (background_scenario_data_df["CH4"].values[-1]) * np.ones(end_year - BACKGROUND_SCENARIO_END_YEAR)
        background_species_quantities["background_CH4"] = np.concatenate((background_species_quantities["background_CH4"],
                                                                    constant_ch4))

        # Background NOx
        constant_nox = (background_scenario_data_df["NOx"].values[-1]) * np.ones(end_year - BACKGROUND_SCENARIO_END_YEAR)
        background_species_quantities["background_NOx"] = np.concatenate((background_species_quantities["background_NOx"],
                                                                    constant_nox))

        # Warning
        warnings.warn(f"Background scenario'{scenario}' has no emission data after 2500. "
                      f"Constant emissions were considered for after 2500.")

    return background_species_quantities