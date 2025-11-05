import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.interpolate import interp1d
from aerocm.metrics.metrics import co2_ipcc_pulse_absolute_metrics
from aerocm.utils.classes import ClimateModel


class IPCCClimateModel(ClimateModel):
    """Class for the IPCC climate model implementation."""

    # --- Default parameters ---
    available_species = [
        "CO2",
        "Contrails",
        "NOx - ST O3 increase",
        "NOx - CH4 decrease and induced",
        "H2O",
        "Soot",
        "Sulfur"
    ]
    available_species_settings = {
        "CO2": {"ratio_erf_rf": {"type": float, "default": 1.0}},
        "Contrails": {"sensitivity_rf": {"type": float, "default": 2.23e-12},
                      "ratio_erf_rf": {"type": float, "default": 0.42},
                      "efficacy_erf": {"type": float, "default": 1.0}},
        "NOx - ST O3 increase": {"sensitivity_rf": {"type": float, "default": 7.64e-12},
                                 "ratio_erf_rf": {"type": float, "default": 1.37},
                                 "efficacy_erf": {"type": float, "default": 1.0}},
        "NOx - CH4 decrease and induced": {"ch4_loss_per_nox": {"type": float, "default": -3.9},
                                           "ratio_erf_rf": {"type": float, "default": 1.18},
                                           "efficacy_erf": {"type": float, "default": 1.0}},
        "H2O": {"sensitivity_rf": {"type": float, "default": 5.2e-15}, "ratio_erf_rf": {"type": float, "default": 1.0},
                "efficacy_erf": {"type": float, "default": 1.0}},
        "Soot": {"sensitivity_rf": {"type": float, "default": 1.0e-10}, "ratio_erf_rf": {"type": float, "default": 1.0},
                 "efficacy_erf": {"type": float, "default": 1.0}},
        "Sulfur": {"sensitivity_rf": {"type": float, "default": -2.0e-11},
                   "ratio_erf_rf": {"type": float, "default": 1.0}, "efficacy_erf": {"type": float, "default": 1.0}}
    }
    available_model_settings = {}

    def run(self, return_df: bool = False) -> dict | pd.DataFrame:
        """Run the IPCC climate model with the assigned input data.

        Parameters
        ----------
        return_df : bool, optional
            If True, returns the results as a pandas DataFrame, by default False.

        Returns
        -------
        output_data : dict
            Dictionary containing the results of the LWE climate model.
        """

        # --- Extract species settings ---
        sensitivity_rf = self.specie_settings.get("sensitivity_rf", None)
        ratio_erf_rf = self.specie_settings["ratio_erf_rf"]
        efficacy_erf = self.specie_settings.get("efficacy_erf", 1.0)
        ch4_loss_per_nox = self.specie_settings.get("ch4_loss_per_nox", 0.0)  # only for NOx - CH4 decrease and induced

        # --- Extract simulation settings ---
        start_year = self.start_year
        end_year = self.end_year
        specie_name = self.specie_name
        specie_inventory = self.specie_inventory
        years = list(range(start_year, end_year + 1))

        # --- Run the IPCC climate model ---
        if specie_name == "CO2":
            equivalent_emissions = (
                    specie_inventory / 10 ** 12
            )  # Conversion from kgCO2 to GtCO2

            co2_molar_mass = 44.01 * 1e-3  # [kg/mol]
            air_molar_mass = 28.97e-3  # [kg/mol]
            atmosphere_total_mass = 5.1352e18  # [kg]
            radiative_efficiency = 1.33e-5  # radiative efficiency [W/m^2/ppb] with AR6 value
            A_co2_unit = (
                    radiative_efficiency
                    * 1e9
                    * air_molar_mass
                    / (co2_molar_mass * atmosphere_total_mass)
            )  # RF per unit mass increase in atmospheric abundance of CO2 [W/m^2/kg]

            A_co2 = A_co2_unit * specie_inventory
            a = [0.2173, 0.2240, 0.2824, 0.2763]
            tau = [0, 394.4, 36.54, 4.304]

            radiative_forcing_from_year = np.zeros(
                (len(specie_inventory), len(specie_inventory))
            )
            # Radiative forcing induced in year j by the species emitted in year i
            for i in range(0, len(specie_inventory)):
                for j in range(0, len(specie_inventory)):
                    if i <= j:
                        radiative_forcing_from_year[i, j] = A_co2[i] * a[0]
                        for k in [1, 2, 3]:
                            radiative_forcing_from_year[i, j] += (
                                    A_co2[i] * a[k] * np.exp(-(j - i) / tau[k])
                            )
            radiative_forcing = np.zeros(len(specie_inventory))
            for k in range(0, len(specie_inventory)):
                radiative_forcing[k] = np.sum(radiative_forcing_from_year[:, k])
            effective_radiative_forcing = radiative_forcing * ratio_erf_rf

        else:
            if specie_name == "NOx - CH4 decrease and induced":
                min_year = min(start_year, 1939)
                max_year = max(end_year, 2051)
                tau_reference_year = [min_year, 1940, 1980, 1994, 2004, 2050, max_year]
                tau_reference_values = [11, 11, 10.1, 10, 9.85, 10.25, 10.25]
                tau_function = interp1d(
                    tau_reference_year, tau_reference_values, kind="linear"
                )
                years = list(range(start_year, end_year + 1))
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
                A_CH4 = A_CH4_unit * ch4_loss_per_nox * specie_inventory
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
                    radiative_forcing[k] = np.sum(radiative_forcing_from_year[:, k])
                effective_radiative_forcing = radiative_forcing * ratio_erf_rf

            else:
                radiative_forcing = sensitivity_rf * specie_inventory
                effective_radiative_forcing = radiative_forcing * ratio_erf_rf

        ## Temperature
        temperature = np.zeros(len(effective_radiative_forcing))
        c = [0.631, 0.429]
        d = [8.4, 409.5]

        if specie_name == "CO2":
            for k in range(0, len(temperature)):
                for ki in range(0, k + 1):
                    term = 0
                    for j in [0, 1]:
                        term += a[0] * c[j] * (1-np.exp((ki-k)/d[j]))
                        for i in [1,2,3]:
                            term += a[i] * tau[i] * c[j] / (tau[i] - d[j]) * (np.exp((ki - k) / tau[i]) - np.exp((ki - k) / d[j]))
                    temperature[k] += A_co2[ki] * term
        elif specie_name == "NOx - CH4 decrease and induced":
            for k in range(0, len(temperature)):
                for ki in range(0, k + 1):
                    term = 0
                    for j in [0, 1]:
                        term += tau[k] * c[j] / (tau[k] - d[j]) * (np.exp((ki - k) / tau[k]) - np.exp((ki - k) / d[j]))
                    temperature[k] += efficacy_erf * (1 + f1 + f2) * A_CH4[ki] * term
        else:
            tau = 1
            for k in range(0, len(temperature)):
                for ki in range(0, k + 1):
                    term = 0
                    for j in [0,1]:
                        term += tau * c[j] / (tau-d[j]) * (np.exp((ki-k)/tau) - np.exp((ki-k)/d[j]))
                    temperature[k] += efficacy_erf * effective_radiative_forcing[ki] * term


        # --- Prepare output ---
        output_data = {
            "radiative_forcing": radiative_forcing,
            "effective_radiative_forcing": effective_radiative_forcing,
            "temperature": temperature
        }

        if return_df:
            output_data = pd.DataFrame(output_data, index=years)
            output_data.index.name = 'Year'

        return output_data