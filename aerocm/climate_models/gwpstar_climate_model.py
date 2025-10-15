import numpy as np
import pandas as pd
from aerocm.metrics.metrics import co2_ipcc_pulse_absolute_metrics
from aerocm.utils.classes import ClimateModel


class GWPStarClimateModel(ClimateModel):
    """GWP* climate model implementation."""

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
        "NOx - ST O3 increase": {"sensitivity_rf": {"type": float, "default": 7.6e-12},
                                 "ratio_erf_rf": {"type": float, "default": 1.37},
                                 "efficacy_erf": {"type": float, "default": 1.0}},
        "NOx - CH4 decrease and induced": {"sensitivity_rf": {"type": float, "default": -6.1e-12},
                                           "ratio_erf_rf": {"type": float, "default": 1.18},
                                           "efficacy_erf": {"type": float, "default": 1.0}},
        "H2O": {"sensitivity_rf": {"type": float, "default": 5.2e-15}, "ratio_erf_rf": {"type": float, "default": 1.0},
                "efficacy_erf": {"type": float, "default": 1.0}},
        "Soot": {"sensitivity_rf": {"type": float, "default": 1.0e-10}, "ratio_erf_rf": {"type": float, "default": 1.0},
                 "efficacy_erf": {"type": float, "default": 1.0}},
        "Sulfur": {"sensitivity_rf": {"type": float, "default": -2.0e-11},
                   "ratio_erf_rf": {"type": float, "default": 1.0}, "efficacy_erf": {"type": float, "default": 1.0}}
    }
    available_model_settings = {"tcre": {"type": float, "default": 0.00045}}

    def run(self, return_df: bool = False) -> dict | pd.DataFrame:
        """Run the GWP* climate model with the assigned input data.

        Parameters
        ----------
        return_df : bool, optional
            If True, returns the results as a pandas DataFrame, by default False.

        Returns
        -------
        output_data : dict
            Dictionary containing the results of the climate model.
        """

        # --- Extract model settings ---
        tcre = self.model_settings["tcre"]

        # --- Extract species settings ---
        sensitivity_rf = self.specie_settings.get("sensitivity_rf", None)
        ratio_erf_rf = self.specie_settings["ratio_erf_rf"]
        efficacy_erf = self.specie_settings.get("efficacy_erf", 1.0)

        # --- Extract simulation settings ---
        start_year = self.start_year
        end_year = self.end_year
        specie_name = self.specie_name
        specie_inventory = self.specie_inventory
        years = list(range(start_year, end_year + 1))

        # --- Run the model ---
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
            radiative_forcing = sensitivity_rf * specie_inventory
            effective_radiative_forcing = radiative_forcing * ratio_erf_rf

            gwpstar_variation_duration = np.nan
            gwpstar_s_coefficient = np.nan

            if (
                    specie_name == "Contrails"
                    or specie_name == "NOx - ST O3 increase"
                    or specie_name == "Soot"
                    or specie_name == "Sulfur"
                    or specie_name == "H2O"
            ):
                gwpstar_variation_duration = 6
                gwpstar_s_coefficient = 0.0

            elif specie_name == "NOx - CH4 decrease and induced":
                gwpstar_variation_duration = 20
                gwpstar_s_coefficient = 0.25

            equivalent_emissions = (
                    gwpstar_equivalent_emissions_function(
                        start_year,
                        end_year,
                        emissions_erf=effective_radiative_forcing,
                        gwpstar_variation_duration=gwpstar_variation_duration,
                        gwpstar_s_coefficient=gwpstar_s_coefficient,
                    )
                    / 10 ** 12
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


def gwpstar_equivalent_emissions_function(
    start_year,
    end_year,
    emissions_erf,
    gwpstar_variation_duration,
    gwpstar_s_coefficient,
):
    # Reference: Smith et al. (2021), https://doi.org/10.1038/s41612-021-00169-8
    # Global
    climate_time_horizon = 100
    (
        agwp_rf_co2,
        agwp_erf_co2,
        aegwp_rf_co2,
        aegwp_erf_co2,
        agtp_co2,
        iagtp_co2,
        atr_co2,
    ) = co2_ipcc_pulse_absolute_metrics(climate_time_horizon)
    co2_agwp_h = agwp_rf_co2

    # g coefficient for GWP*
    if gwpstar_s_coefficient == 0:
        g_coefficient = 1
    else:
        g_coefficient = (
            1 - np.exp(-gwpstar_s_coefficient / (1 - gwpstar_s_coefficient))
        ) / gwpstar_s_coefficient

    # Main
    emissions_erf_variation = np.zeros(end_year - start_year + 1)
    for k in range(start_year, end_year + 1):
        if k - start_year >= gwpstar_variation_duration:
            emissions_erf_variation[k - start_year] = (
                emissions_erf[k - start_year]
                - emissions_erf[k - gwpstar_variation_duration - start_year]
            ) / gwpstar_variation_duration
        else:
            emissions_erf_variation[k - start_year] = (
                emissions_erf[k - start_year] / gwpstar_variation_duration
            )
    emissions_equivalent_emissions = np.zeros(end_year - start_year + 1)
    for k in range(start_year, end_year + 1):
        emissions_equivalent_emissions[k - start_year] = (
            g_coefficient
            * (1 - gwpstar_s_coefficient)
            * climate_time_horizon
            / co2_agwp_h
            * emissions_erf_variation[k - start_year]
        ) + g_coefficient * gwpstar_s_coefficient / co2_agwp_h * emissions_erf[
            k - start_year
        ]

    return emissions_equivalent_emissions