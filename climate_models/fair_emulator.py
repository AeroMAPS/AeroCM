from climate_data import RCP
import os.path as pth
import numpy as np
import pandas as pd
from fair import FAIR
from fair.interface import fill, initialise
from functions.functions import AbsoluteMetricsPulseDefaultCO2


def GWPStarEquivalentEmissionsFunction(start_year, end_year, emissions_erf, gwpstar_variation_duration, gwpstar_s_coefficient):
    # Reference: Smith et al. (2021), https://doi.org/10.1038/s41612-021-00169-8
    # Global
    climate_time_horizon = 100
    rf_co2, agwp_co2, aegwp_co2, temp_co2, agtp_co2, iagtp_co2, atr_co2 = AbsoluteMetricsPulseDefaultCO2(climate_time_horizon, 1)
    co2_agwp_h = agwp_co2

    # g coefficient for GWP*
    if gwpstar_s_coefficient == 0:
        g_coefficient = 1
    else:
        g_coefficient = (
                                1 - np.exp(-gwpstar_s_coefficient / (1 - gwpstar_s_coefficient))
                        ) / gwpstar_s_coefficient

    # Main
    emissions_erf_variation = np.zeros(end_year-start_year+1)
    for k in range(start_year, end_year + 1):
        if k - start_year >= gwpstar_variation_duration:
            emissions_erf_variation[k-start_year] = (
                                                                        emissions_erf[k-start_year] - emissions_erf[
                                                                    k - gwpstar_variation_duration - start_year]
                                                                ) / gwpstar_variation_duration
        else:
            emissions_erf_variation[k-start_year] = (
                    emissions_erf[k-start_year] / gwpstar_variation_duration
            )
    emissions_equivalent_emissions = np.zeros(end_year-start_year+1)
    for k in range(start_year, end_year + 1):
        emissions_equivalent_emissions[k-start_year] = (
                                                                           g_coefficient
                                                                           * (1 - gwpstar_s_coefficient)
                                                                           * climate_time_horizon
                                                                           / co2_agwp_h
                                                                           * emissions_erf_variation[k-start_year]
                                                                   ) + g_coefficient * gwpstar_s_coefficient / co2_agwp_h * \
                                                                   emissions_erf[k-start_year]

    return emissions_equivalent_emissions




def RunFair(start_year, end_year, background_species_quantities, studied_species='None', studied_species_quantities=0):
    # Creation of FaIR instance
    f = FAIR()

    # Definition of time horizon, scenarios, configs
    f.define_time(start_year, end_year, 1)
    f.define_scenarios(["central"])
    f.define_configs(["central"])
    # f.define_configs(["high", "central", "low"])

    # Definition of species and properties
    species = [
        "CO2",  # Includes world emissions, aviation emissions, and equivalent emissions for NOx effects (except ST O3)
        "World CH4",
        "Aviation contrails",
        "Aviation NOx ST O3 increase",
        "Aviation H2O",
        "Aviation sulfur",
        "Aviation soot",
        "Aviation aerosols",
    ]
    properties = {
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
            "aerosol_chemistry_from_concentration": True,  # we treat methane as a reactive gas
        },
        "Aviation contrails": {
            "type": "contrails",
            "input_mode": "forcing",
            "greenhouse_gas": False,
            "aerosol_chemistry_from_emissions": False,
            "aerosol_chemistry_from_concentration": False,
        },
        "Aviation NOx ST O3 increase": {
            "type": "ozone",
            "input_mode": "forcing",
            "greenhouse_gas": False,
            "aerosol_chemistry_from_emissions": False,
            "aerosol_chemistry_from_concentration": False,
        },
        "Aviation H2O": {
            "type": "h2o stratospheric",
            "input_mode": "forcing",
            "greenhouse_gas": False,
            "aerosol_chemistry_from_emissions": False,
            "aerosol_chemistry_from_concentration": False,
        },
        "Aviation sulfur": {
            "type": "sulfur",
            "input_mode": "emissions",
            "greenhouse_gas": False,
            "aerosol_chemistry_from_emissions": True,
            "aerosol_chemistry_from_concentration": False,
        },
        "Aviation soot": {
            "type": "black carbon",
            "input_mode": "emissions",
            "greenhouse_gas": False,
            "aerosol_chemistry_from_emissions": True,
            "aerosol_chemistry_from_concentration": False,
        },
        # Dedicated specie for aerosols
        "Aviation aerosols": {
            "type": "ari",
            "input_mode": "calculated",
            "greenhouse_gas": False,
            "aerosol_chemistry_from_emissions": False,
            "aerosol_chemistry_from_concentration": False,
        },
    }
    f.define_species(species, properties)

    # Definition of run options
    f.ghg_method = "leach2021"
    f.aci_method = "myhre1998"

    # Creation of input and output data
    f.allocate()

    # Filling species quantities
    if studied_species == 'Aviation NOx':
        total_CO2 = (
                background_species_quantities[0][1: end_year - start_year + 1]
                + studied_species_quantities[1][1: end_year - start_year + 1]
                + studied_species_quantities[2][1: end_year - start_year + 1]
                + studied_species_quantities[3][1: end_year - start_year + 1]
        )
        fill(
            f.emissions,
            total_CO2,
            specie="CO2",
            config=f.configs[0],
            scenario=f.scenarios[0],
        )
        fill(
            f.emissions,
            background_species_quantities[1][1 : end_year - start_year + 1],
            specie="World CH4",
            config=f.configs[0],
            scenario=f.scenarios[0],
        )
        fill(
            f.forcing, 0, specie="Aviation contrails", config=f.configs[0], scenario=f.scenarios[0]
        )
        fill(
            f.forcing,
            studied_species_quantities[0],
            specie="Aviation NOx ST O3 increase",
            config=f.configs[0],
            scenario=f.scenarios[0],
        )
        fill(f.forcing, 0, specie="Aviation H2O", config=f.configs[0], scenario=f.scenarios[0])
        fill(f.emissions, 0, specie="Aviation sulfur", config=f.configs[0], scenario=f.scenarios[0])
        fill(f.emissions, 0, specie="Aviation soot", config=f.configs[0], scenario=f.scenarios[0])

    else:
        if studied_species == "Aviation CO2" or studied_species == "Aviation NOx LT O3 decrease" or studied_species == "Aviation NOx CH4 decrease" or studied_species == "Aviation NOx SWV decrease":
            total_CO2 = (
                background_species_quantities[0][1 : end_year - start_year + 1]
                + studied_species_quantities[1 : end_year - start_year + 1]
            )
        else:
            total_CO2 = background_species_quantities[0][1 : end_year - start_year + 1]
        fill(f.emissions, total_CO2, specie="CO2", config=f.configs[0], scenario=f.scenarios[0])
        fill(
            f.emissions,
            background_species_quantities[1][1 : end_year - start_year + 1],
            specie="World CH4",
            config=f.configs[0],
            scenario=f.scenarios[0],
        )
        if studied_species == "Aviation contrails":
            fill(
                f.forcing,
                studied_species_quantities,
                specie="Aviation contrails",
                config=f.configs[0],
                scenario=f.scenarios[0],
            )
        else:
            fill(
                f.forcing,
                0,
                specie="Aviation contrails",
                config=f.configs[0],
                scenario=f.scenarios[0],
            )
        if studied_species == "Aviation NOx ST O3 increase":
            fill(
                f.forcing,
                studied_species_quantities,
                specie="Aviation NOx ST O3 increase",
                config=f.configs[0],
                scenario=f.scenarios[0],
            )
        else:
            fill(
                f.forcing,
                0,
                specie="Aviation NOx ST O3 increase",
                config=f.configs[0],
                scenario=f.scenarios[0],
            )
        if studied_species == "Aviation H2O":
            fill(
                f.forcing,
                studied_species_quantities,
                specie="Aviation H2O",
                config=f.configs[0],
                scenario=f.scenarios[0],
            )
        else:
            fill(f.forcing, 0, specie="Aviation H2O", config=f.configs[0], scenario=f.scenarios[0])
        if studied_species == "Aviation sulfur":
            fill(
                f.emissions,
                studied_species_quantities[1 : end_year - start_year + 1],
                specie="Aviation sulfur",
                config=f.configs[0],
                scenario=f.scenarios[0],
            )
        else:
            fill(
                f.emissions,
                0,
                specie="Aviation sulfur",
                config=f.configs[0],
                scenario=f.scenarios[0],
            )
        if studied_species == "Aviation soot":
            fill(
                f.emissions,
                studied_species_quantities[1 : end_year - start_year + 1],
                specie="Aviation soot",
                config=f.configs[0],
                scenario=f.scenarios[0],
            )
        else:
            fill(
                f.emissions, 0, specie="Aviation soot", config=f.configs[0], scenario=f.scenarios[0]
            )

    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    # Filling climate configs
    # fill(f.climate_configs["ocean_heat_transfer"], [1.1, 1.6, 0.9], config="central")
    # fill(f.climate_configs["ocean_heat_capacity"], [8, 14, 100], config="central")
    # fill(f.climate_configs["deep_ocean_efficacy"], 1.1, config="central")
    # Corresponds to a "low" configuration on FaIR
    fill(f.climate_configs["ocean_heat_transfer"], [1.7, 2.0, 1.1], config="central")
    fill(f.climate_configs["ocean_heat_capacity"], [6, 11, 75], config="central")
    fill(f.climate_configs["deep_ocean_efficacy"], 0.8, config="central")

    # Filling species configs
    for specie in species:
        if specie == "CO2":
            fill(
                f.species_configs["partition_fraction"],
                [0.2173, 0.2240, 0.2824, 0.2763],
                specie="CO2",
            )
            fill(
                f.species_configs["unperturbed_lifetime"], [1e9, 394.4, 36.54, 4.304], specie="CO2"
            )
            fill(f.species_configs["baseline_concentration"], 278.3, specie="CO2")
            fill(f.species_configs["forcing_reference_concentration"], 278.3, specie="CO2")
            fill(f.species_configs["molecular_weight"], 44.009, specie="CO2")
            fill(
                f.species_configs["greenhouse_gas_radiative_efficiency"],
                1.3344985680386619e-05,
                specie="CO2",
            )
            f.calculate_iirf0()
            f.calculate_g()
            f.calculate_concentration_per_emission()
            fill(f.species_configs["iirf_0"], 29, specie="CO2")
            fill(f.species_configs["iirf_airborne"], [0.000819 * 2], specie="CO2")
            fill(f.species_configs["iirf_uptake"], [0.00846 * 2], specie="CO2")
            fill(f.species_configs["iirf_temperature"], [8], specie="CO2")
            fill(f.species_configs["aci_scale"], -2.09841432)

        if specie == "World CH4":
            fill(f.species_configs["partition_fraction"], [1, 0, 0, 0], specie=specie)
            fill(f.species_configs["unperturbed_lifetime"], 8.25, specie=specie)
            fill(f.species_configs["baseline_concentration"], 729, specie=specie)  # ppb
            fill(f.species_configs["forcing_reference_concentration"], 729, specie=specie)
            fill(f.species_configs["molecular_weight"], 16.043, specie=specie)
            fill(
                f.species_configs["greenhouse_gas_radiative_efficiency"],
                0.00038864402860869495,
                specie=specie,
            )
            f.calculate_iirf0()
            f.calculate_g()
            f.calculate_concentration_per_emission()
            fill(f.species_configs["iirf_airborne"], 0.00032, specie=specie)
            fill(f.species_configs["iirf_uptake"], 0, specie=specie)
            fill(f.species_configs["iirf_temperature"], -0.3, specie=specie)
            fill(
                f.species_configs["erfari_radiative_efficiency"],
                -0.002653 / 1023.2219696044921,
                specie=specie,
            )  # W m-2 ppb-1
            fill(f.species_configs["aci_scale"], -2.09841432)

        if specie == "Aviation sulfur":
            erf_aci_sulfur = 0.0
            fill(
                f.species_configs["erfari_radiative_efficiency"],
                -0.0199 + erf_aci_sulfur,
                specie=specie,
            )  # W m-2 MtSO2-1 yr
            fill(f.species_configs["aci_shape"], 0.0, specie=specie)

        if specie == "Aviation soot":
            erf_aci_BC = 0.0
            fill(
                f.species_configs["erfari_radiative_efficiency"], 0.1007 + erf_aci_BC, specie=specie
            )  # W m-2 MtC-1 yr
            fill(f.species_configs["aci_shape"], 0.0, specie=specie)

    # Run
    f.run()

    return (
        f.temperature.loc[dict(config=f.configs[0], layer=0)].data,
        f.forcing_sum.loc[dict(config=f.configs[0])].data,
    )



def BackgroundSpeciesQuantities(start_year, end_year, rcp):
    if rcp == 'RCP26':
        rcp_data_path = pth.join(RCP.__path__[0], "RCP26.csv")
    elif rcp == 'RCP45':
        rcp_data_path = pth.join(RCP.__path__[0], "RCP45.csv")
    elif rcp == 'RCP60':
        rcp_data_path = pth.join(RCP.__path__[0], "RCP60.csv")
    elif rcp == 'RCP85':
        rcp_data_path = pth.join(RCP.__path__[0], "RCP85.csv")
    rcp_data_df = pd.read_csv(rcp_data_path)

    background_species_quantities = np.zeros((2, end_year - start_year + 1))

    ### World CO2
    background_species_quantities[0] = (
            (
                    rcp_data_df["FossilCO2"][0: end_year - start_year + 1].values
                    + rcp_data_df["OtherCO2"][0: end_year - start_year + 1].values
            )
            * 44
            / 12
    )  # Conversion from GtC to GtCO2

    ## World CH4
    background_species_quantities[1] = rcp_data_df["CH4"][0: end_year - start_year + 1].values  # Unit: MtCH4

    return background_species_quantities


def FaIRClimateModel(start_year, end_year, background_species_quantities, emission_profile, studied_species, sensitivity_erf, ratio_erf_rf):

    if studied_species == 'Aviation CO2':
        studied_species_quantities = emission_profile / 10**12  # Conversion from kgCO2 to GtCO2
    elif studied_species == 'Aviation soot':
        studied_species_quantities = emission_profile / 10**9  # Conversion from kgSO2 to MtSO2
    elif studied_species == 'Aviation sulfur':
        studied_species_quantities = emission_profile / 10**9  # Conversion from kgBC to MtBC
    elif studied_species == 'Aviation contrails':
        erf = sensitivity_erf * emission_profile
        studied_species_quantities = erf  # W/m2
    elif studied_species == 'Aviation H2O':
        erf = sensitivity_erf * emission_profile
        studied_species_quantities = erf  # W/m2
    elif studied_species == 'Aviation NOx ST O3 increase':
        erf = sensitivity_erf * emission_profile
        studied_species_quantities = erf  # W/m2
    elif studied_species == 'Aviation NOx LT O3 decrease':
        erf = sensitivity_erf * emission_profile
        studied_species_quantities = GWPStarEquivalentEmissionsFunction(
            start_year,
            end_year,
            emissions_erf=erf,
            gwpstar_variation_duration=20,
            gwpstar_s_coefficient=0.25,
        ) / 10**12 # Conversion from kgCO2 to GtCO2
    elif studied_species == 'Aviation NOx CH4 decrease':
        erf = sensitivity_erf * emission_profile
        studied_species_quantities = GWPStarEquivalentEmissionsFunction(
            start_year,
            end_year,
            emissions_erf=erf,
            gwpstar_variation_duration=20,
            gwpstar_s_coefficient=0.25,
        ) / 10**12 # Conversion from kgCO2 to GtCO2
    elif studied_species == 'Aviation NOx SWV decrease':
        erf = sensitivity_erf * emission_profile
        studied_species_quantities = GWPStarEquivalentEmissionsFunction(
            start_year,
            end_year,
            emissions_erf=erf,
            gwpstar_variation_duration=20,
            gwpstar_s_coefficient=0.25,
        ) / 10**12 # Conversion from kgCO2 to GtCO2
    elif studied_species == 'Aviation NOx':
        erf = np.zeros((4, len(emission_profile)))
        studied_species_quantities = np.zeros((4, len(emission_profile)))
        for k in range(0,len(erf)):
            erf[k] = sensitivity_erf[k] * emission_profile
            if k == 0:
                studied_species_quantities[k] = erf[k]# W/m2
            else:
                studied_species_quantities[k] = GWPStarEquivalentEmissionsFunction(
                    start_year,
                    end_year,
                    emissions_erf=erf[k],
                    gwpstar_variation_duration=20,
                    gwpstar_s_coefficient=0.25,
                ) / 10**12 # Conversion from kgCO2 to GtCO2

    temperature_with_species, effective_radiative_forcing_with_species = RunFair(start_year, end_year, background_species_quantities, studied_species, studied_species_quantities)
    temperature_without_species, effective_radiative_forcing_without_species = RunFair(start_year, end_year, background_species_quantities, studied_species = 'None')
    temperature = temperature_with_species - temperature_without_species
    effective_radiative_forcing = effective_radiative_forcing_with_species - effective_radiative_forcing_without_species
    if studied_species == 'Aviation NOx':
        radiative_forcing = effective_radiative_forcing / np.mean(ratio_erf_rf)
    else:
        radiative_forcing = effective_radiative_forcing / ratio_erf_rf

    return radiative_forcing, effective_radiative_forcing, temperature