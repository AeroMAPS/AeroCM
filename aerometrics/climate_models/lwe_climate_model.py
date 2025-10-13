import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.interpolate import interp1d
from aerometrics.metrics.metrics import co2_ipcc_pulse_absolute_metrics
from aerometrics.utils.classes import ClimateModel


class LWEClimateModel(ClimateModel):
    """Class for the Linear Warming Equivalent (LWE) climate model implementation."""

    # --- Variables for validation ---
    available_species = [
        "CO2",
        "Contrails",
        "NOx - ST O3 increase",
        "NOx - CH4 decrease and induced",
        "Soot",
        "Sulfur",
        "H2O"
    ]
    available_species_settings = {
        "CO2": {"ratio_erf_rf": float},
        "Contrails": {"sensitivity_rf": float, "ratio_erf_rf": float, "efficacy_erf": float},
        "NOx - ST O3 increase": {"sensitivity_rf": float, "ratio_erf_rf": float, "efficacy_erf": float},
        "NOx - CH4 decrease and induced": {"ch4_loss_per_nox": float, "ratio_erf_rf": float, "efficacy_erf": float},
        "Soot": {"sensitivity_rf": float, "ratio_erf_rf": float, "efficacy_erf": float},
        "Sulfur": {"sensitivity_rf": float, "ratio_erf_rf": float, "efficacy_erf": float},
        "H2O": {"sensitivity_rf": float, "ratio_erf_rf": float, "efficacy_erf": float},
    }
    available_model_settings = {"tcre": float}

    def run(self, return_df: bool = False) -> dict | pd.DataFrame:
        """Run the LWE climate model with the assigned input data.

        Parameters
        ----------
        return_df : bool, optional
            If True, returns the results as a pandas DataFrame, by default False.

        Returns
        -------
        output_data : dict
            Dictionary containing the results of the LWE climate model.
        """

        # --- Extract model settings ---
        tcre = self.model_settings["tcre"]

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

        # --- Run the LWE climate model ---
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

            size = end_year - start_year + 1
            F_co2 = np.zeros((size, size))

            # Old version for filling F_CO2 (long calculation time)
            # for i in range(0, size):
            #     for j in range(0, size):
            #         if i > j:
            #             start = time.time()
            #             agwp_rf_co2_1, *rest = co2_ipcc_pulse_absolute_metrics(i - j + 1)
            #             print("Matrix creation:", time.time() - start)
            #             agwp_rf_co2, *rest = co2_ipcc_pulse_absolute_metrics(i - j)
            #             F_co2[i, j] = agwp_rf_co2_1 - agwp_rf_co2
            #
            #         elif i == j:
            #             agwp_rf_co2, *rest = co2_ipcc_pulse_absolute_metrics(1)
            #             F_co2[i, j] = agwp_rf_co2

            agwp_data = {}
            for delta in range(1, size + 1):
                agwp, *rest = co2_ipcc_pulse_absolute_metrics(delta)
                agwp_data[delta] = agwp

            for i in range(size):
                for j in range(size):
                    delta = i - j
                    if delta > 0:
                        F_co2[i, j] = agwp_data[delta + 1] - agwp_data[delta]
                    elif delta == 0:
                        F_co2[i, j] = agwp_data[1]

            # Inverting F_CO2 by using solve_triangular function (more efficient than np.linalg.inv)
            Identity = np.eye(F_co2.shape[0])
            F_co2_inv = solve_triangular(F_co2, Identity, lower=True)

            equivalent_emissions = (
                    np.dot(F_co2_inv, effective_radiative_forcing) / 10 ** 12
            )  # Conversion from kgCO2-we to GtCO2-we

        cumulative_equivalent_emissions = np.zeros(len(specie_inventory))
        cumulative_equivalent_emissions[0] = equivalent_emissions[0]
        for k in range(1, len(cumulative_equivalent_emissions)):
            cumulative_equivalent_emissions[k] = (
                    cumulative_equivalent_emissions[k - 1] + equivalent_emissions[k]
            )
        temperature = tcre * cumulative_equivalent_emissions * efficacy_erf

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