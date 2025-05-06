from climate_data import RCP
import os.path as pth
import numpy as np
import pandas as pd
from fair import FAIR
from fair.interface import fill, initialise


def run_fair(start_year, end_year, species_studied, studied_species_quantities, params):

    # params = {
    #     "background_species_quantities": background_species_quantities,
    #     "sensitivity_erf": sensitivity_erf,
    #     "ratio_erf_rf": ratio_erf_rf,
    #     "efficacy_erf": efficacy_erf,
    # }

    background_species_quantities = params["background_species_quantities"]
    efficacy_erf = params["efficacy_erf"]

    # Creation of FaIR instance
    f = FAIR()

    # Definition of time horizon, scenarios, configs
    f.define_time(start_year, end_year, 1)
    f.define_scenarios(["central"])
    f.define_configs(["central"])
    # f.define_configs(["high", "central", "low"])

    # Definition of species and properties
    species = [
        "CO2",  # Includes world and aviation emissions
        "World CH4",
        "Aviation contrails",
        "Aviation NOx - ST O3 increase",
        "Aviation NOx - CH4 decrease and induced",
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
        "Aviation NOx - ST O3 increase": {
            "type": "ozone",
            "input_mode": "forcing",
            "greenhouse_gas": False,
            "aerosol_chemistry_from_emissions": False,
            "aerosol_chemistry_from_concentration": False,
        },
        "Aviation NOx - CH4 decrease and induced": {
            "type": "unspecified",
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
    if species_studied == "Aviation CO2":
        total_CO2 = (
            background_species_quantities[0][1 : end_year - start_year + 1]
            + studied_species_quantities[1 : end_year - start_year + 1]
        )
    else:
        total_CO2 = background_species_quantities[0][1 : end_year - start_year + 1]

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
    if species_studied == "Aviation contrails":
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
    if species_studied == "Aviation NOx - ST O3 increase":
        fill(
            f.forcing,
            studied_species_quantities,
            specie="Aviation NOx - ST O3 increase",
            config=f.configs[0],
            scenario=f.scenarios[0],
        )
    else:
        fill(
            f.forcing,
            0,
            specie="Aviation NOx - ST O3 increase",
            config=f.configs[0],
            scenario=f.scenarios[0],
        )
    if species_studied == "Aviation NOx - CH4 decrease and induced":
        fill(
            f.forcing,
            studied_species_quantities,
            specie="Aviation NOx - CH4 decrease and induced",
            config=f.configs[0],
            scenario=f.scenarios[0],
        )
    else:
        fill(
            f.forcing,
            0,
            specie="Aviation NOx - CH4 decrease and induced",
            config=f.configs[0],
            scenario=f.scenarios[0],
        )
    if species_studied == "Aviation H2O":
        fill(
            f.forcing,
            studied_species_quantities,
            specie="Aviation H2O",
            config=f.configs[0],
            scenario=f.scenarios[0],
        )
    else:
        fill(
            f.forcing,
            0,
            specie="Aviation H2O",
            config=f.configs[0],
            scenario=f.scenarios[0],
        )
    if species_studied == "Aviation sulfur":
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
    if species_studied == "Aviation soot":
        fill(
            f.emissions,
            studied_species_quantities[1 : end_year - start_year + 1],
            specie="Aviation soot",
            config=f.configs[0],
            scenario=f.scenarios[0],
        )
    else:
        fill(
            f.emissions,
            0,
            specie="Aviation soot",
            config=f.configs[0],
            scenario=f.scenarios[0],
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
                f.species_configs["unperturbed_lifetime"],
                [1e9, 394.4, 36.54, 4.304],
                specie="CO2",
            )
            fill(f.species_configs["baseline_concentration"], 278.3, specie="CO2")
            fill(
                f.species_configs["forcing_reference_concentration"],
                278.3,
                specie="CO2",
            )
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
            fill(
                f.species_configs["forcing_reference_concentration"], 729, specie=specie
            )
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

        if specie == "Aviation contrails":
            fill(f.species_configs["forcing_efficacy"], efficacy_erf, specie=specie)

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
                f.species_configs["erfari_radiative_efficiency"],
                0.1007 + erf_aci_BC,
                specie=specie,
            )  # W m-2 MtC-1 yr
            fill(f.species_configs["aci_shape"], 0.0, specie=specie)

    # Run
    f.run()

    return (
        f.temperature.loc[dict(config=f.configs[0], layer=0)].data,
        f.forcing_sum.loc[dict(config=f.configs[0])].data,
    )


def background_species_quantities_function(start_year, end_year, rcp):
    if rcp == "RCP26":
        rcp_data_path = pth.join(RCP.__path__[0], "RCP26.csv")
    elif rcp == "RCP45":
        rcp_data_path = pth.join(RCP.__path__[0], "RCP45.csv")
    elif rcp == "RCP60":
        rcp_data_path = pth.join(RCP.__path__[0], "RCP60.csv")
    elif rcp == "RCP85":
        rcp_data_path = pth.join(RCP.__path__[0], "RCP85.csv")
    rcp_data_df = pd.read_csv(rcp_data_path)

    background_species_quantities = np.zeros((2, end_year - start_year + 1))

    ### World CO2
    background_species_quantities[0] = (
        (
            rcp_data_df["FossilCO2"][0 : end_year - start_year + 1].values
            + rcp_data_df["OtherCO2"][0 : end_year - start_year + 1].values
        )
        * 44
        / 12
    )  # Conversion from GtC to GtCO2

    ## World CH4
    background_species_quantities[1] = rcp_data_df["CH4"][
        0 : end_year - start_year + 1
    ].values  # Unit: MtCH4

    return background_species_quantities


def species_fair_climate_model(
    start_year, end_year, species_studied, species_quantities, params
):

    # params = {
    #     "background_species_quantities": background_species_quantities,
    #     "sensitivity_erf": sensitivity_erf,
    #     "ratio_erf_rf": ratio_erf_rf,
    #     "efficacy_erf": efficacy_erf,
    # }

    sensitivity_erf = params["sensitivity_erf"]
    ratio_erf_rf = params["ratio_erf_rf"]

    if species_studied == "Aviation CO2":
        studied_species_quantities = (
            species_quantities / 10**12
        )  # Conversion from kgCO2 to GtCO2
    elif species_studied == "Aviation soot":
        studied_species_quantities = (
            species_quantities / 10**9
        )  # Conversion from kgSO2 to MtSO2
    elif species_studied == "Aviation sulfur":
        studied_species_quantities = (
            species_quantities / 10**9
        )  # Conversion from kgBC to MtBC
    elif species_studied == "Aviation contrails":
        erf = sensitivity_erf * species_quantities
        studied_species_quantities = erf  # W/m2
    elif species_studied == "Aviation H2O":
        erf = sensitivity_erf * species_quantities
        studied_species_quantities = erf  # W/m2
    elif species_studied == "Aviation NOx - ST O3 increase":
        erf = sensitivity_erf * species_quantities
        studied_species_quantities = erf  # W/m2
    elif species_studied == "Aviation NOx - CH4 decrease and induced":
        tau = 11.8
        A_CH4_unit = 5.7e-4
        A_CH4 = A_CH4_unit * sensitivity_erf * species_quantities
        f1 = 0.5  # Indirect effect on ozone
        f2 = 0.15  # Indirect effect on stratospheric water
        effective_radiative_forcing_from_year = np.zeros(
            (len(species_quantities), len(species_quantities))
        )
        # Effective radiative forcing induced in year j by the species emitted in year i
        for i in range(0, len(species_quantities)):
            for j in range(0, len(species_quantities)):
                if i <= j:
                    effective_radiative_forcing_from_year[i, j] = (
                        (1 + f1 + f2) * A_CH4[i] * np.exp(-(j - i) / tau)
                    )
        effective_radiative_forcing = np.zeros(len(species_quantities))
        for k in range(0, len(species_quantities)):
            effective_radiative_forcing[k] = np.sum(
                effective_radiative_forcing_from_year[:, k]
            )
        studied_species_quantities = effective_radiative_forcing  # W/m2

    temperature_with_species, effective_radiative_forcing_with_species = run_fair(
        start_year, end_year, species_studied, studied_species_quantities, params
    )
    temperature_without_species, effective_radiative_forcing_without_species = run_fair(
        start_year,
        end_year,
        species_studied="None",
        studied_species_quantities=0,
        params=params,
    )
    temperature = temperature_with_species - temperature_without_species
    effective_radiative_forcing = (
        effective_radiative_forcing_with_species
        - effective_radiative_forcing_without_species
    )
    radiative_forcing = effective_radiative_forcing / ratio_erf_rf

    return radiative_forcing, effective_radiative_forcing, temperature
