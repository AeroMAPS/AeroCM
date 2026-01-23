""" 
Utility functions used to prepare data for OAC, run OAC and handle outputs
"""
import numpy as np
import xarray as xr
from pandas import read_csv
import os


def read_nc(nc_file):
    """
    Parameters:
    ----------
    nc_file: file path for nc file to be read

    Returns
    -------
    output_data : dict
        Dictionary containing lat lon, altitude coordinates and the spc emission values
    """
    xrds = xr.open_dataset(nc_file)
    lat = xrds['lat'].values
    lon = xrds['lon'].values
    pres = xrds['plev'].values
    co2 = xrds['CO2'].values
    h2o = xrds['H2O'].values
    nox = xrds['NOx'].values
    dist = xrds['distance'].values

    return {'lat':lat, 'lon': lon, 'pres':pres, 'CO2': co2, 'NOx': nox, 'H2O': h2o, 'Contrails' : dist}

def scaled_emissions_spc(spc, agg, nc_file):
    """function to return scaled nc data, needs to be adpated for any species

    Parameters
    ----------
    spc : string
        species to be rescaled

    agg : float
        desired global spc emission

    nc_file: string
        file path of nc file to be scaled

    Returns
    -------
    scl_spc : list
        scaled spc emissions

    """
    spc_val = read_nc(nc_file)[spc]
    tot_spc = np.sum(spc_val)
    scf = agg/tot_spc
    scl_spc = spc_val* scf
    return scl_spc

def scaled_emissions_to_nc(input_nc, output_nc, spc, agg, year):
    """ takes in the spc emissions and converts into individual ncdf file

    Parameters
    ----------
    input_nc : str
        base/reference nc file path
    output_nc : str
        name of the output nc 
    spc : string
        species to be rescaled
    agg : float
        desired global spc emission
    year : float
        year of emission inventory    
    
    Returns 
    -------
    output file : nc file
        nc file of emission stored in output file (for now called inputs)    
    """
    os.makedirs("oac_inputs", exist_ok=True)
    xrds = xr.open_dataset(input_nc)
    scaled_spc = scaled_emissions_spc(spc, agg, input_nc)

    spc_list = ['CO2', 'H2O','NOx','distance']

    #new_ds = xrds.copy()
    if spc == "Contrails":
        spc = "distance"
        spc_list.remove(spc)
        new_ds = xrds.drop_vars(spc_list)
        new_ds[spc] = (xrds[spc].dims, scaled_spc)
        new_ds.attrs['Inventory_Year'] = year
        new_ds[spc].attrs['long_name'] = "distance flown"
        new_ds[spc].attrs['units'] = 'km'
    else:     
        spc_list.remove(spc)
        new_ds = xrds.drop_vars(spc_list)
        new_ds[spc] = (xrds[spc].dims, scaled_spc)
        new_ds.attrs['Inventory_Year'] = year
        new_ds[spc].attrs['long_name'] = spc
        new_ds[spc].attrs['units'] = 'kg'
     

    output_path = os.path.join("oac_inputs", output_nc)
    new_ds.to_netcdf(output_path)
    print(f"Saved scaled emissions to {output_path}")

def time_norm_ncdf(years, species_list, species_inventory, nc_name):
    """ Function to generate time normalisation evolution netCDF file with the emission indices for each species.
    
    Parameters
    ----------
    years : list
        list of simulation years.
    species_list : list
        list of species being simulated.
    species_inventory : dict
        dictionary of simulated species with their respective emissions.
    nc_name : str
        name of time evolution netCDF file.    

    Returns
    -------
    {nc_name}.nc : netCDF file
        time evoltuion file.       

    """

    os.makedirs("time_evo", exist_ok= True)
    # --- Determining emission indices --- 
    fuel = species_inventory["CO2"] / 3.15 * 1e-9
    EI_CO2 = np.array([3.15] * len(years))
    
    #EI_NOx = None
    EI_H2O = species_inventory["H2O"] / (fuel *1e9)
    dis_per_fuel = species_inventory["Contrails"] / (fuel *1e9)
    #if "NOx" in species_list:    
    #    EI_NOx = (species_inventory["NOx"] * 27.79543563) / (fuel *1e9) * 0.03597713
    

    ds = xr.Dataset(
        data_vars={
            "fuel":        (("time",), np.asarray(fuel, dtype=np.float32)),
            "EI_CO2":      (("time",), np.asarray(EI_CO2, dtype=np.float32)),
            #"EI_NOx":      (("time",), np.asarray(EI_NOx, dtype=np.float32)),
            "EI_H2O":      (("time",), np.asarray(EI_H2O, dtype=np.float32)),
            "dis_per_fuel":(("time",), np.asarray(dis_per_fuel, dtype=np.float32))
            
        },
        coords={
            "time": ("time", np.asarray(years, dtype=np.int32))
        },
        attrs={
            "Title":       "Time normalization ",
            "Convention":  "CF-XXX",
            "Type":        "norm",
            "Author":      "Abhigyan Prakash based on OAC example",
        }
    )
    
    
    ds["fuel"].attrs.update(units="Tg yr-1", long_name="fuel mass")
    ds["EI_CO2"].attrs.update(units="", long_name="CO2 emission index")
    #ds["EI_NOx"].attrs.update(units="", long_name="NOx emission index")
    ds["EI_H2O"].attrs.update(units="", long_name="H2O emission index")
    ds["dis_per_fuel"].attrs.update(units="km kg-1", long_name="distance per fuel")
    ds["time"].attrs.update(long_name="year")

    encoding = {var: {"dtype": "float32", "zlib": True, "complevel": 4} for var in ds.data_vars}
    encoding["time"] = {"dtype": "int32"}

    outpath = f"time_evo/{nc_name}"
    ds.to_netcdf(outpath, encoding=encoding)

    print("Saved new NetCDF to:", outpath)
    print("Dataset summary:")
    print(ds)

# Function to read and generate species_inventory. Can be used for every climate model
# Note: file path only correct when bein called in the climate_models folder
def load_species_inventory(start_year, end_year, climate_model, csv_path="../climate_data/aviation_emissions_data.csv"):
    """
    Load species inventory for a given year range from the aviation emissions CSV.

    Parameters
    ----------
    start_year : int
        First simulation year.
    end_year : int
        Last simulation year.
    climate_model : 
        Specify which climate model it is preparing the data for (right now its only specified to OAC)
    csv_path : str
        Path to the aviation emissions CSV.

    Returns
    -------
    dict
        Dictionary containing species emission time series for the given period.
    """

    
    df = read_csv(csv_path, delimiter=";")
    data = df.values

    years = data[:, 0]
    co2 = data[:, 1]
    nox = data[:, 2]
    h2o = data[:, 3]
    soot = data[:, 4]
    sulfur = data[:, 5]
    distance = data[:, 6]

    mask = (years >= start_year) & (years <= end_year)

    co2_emissions = co2[mask]
    nox_emissions = nox[mask]
    h2o_emissions = h2o[mask]
    soot_emissions = soot[mask]
    sulfur_emissions = sulfur[mask]
    distance_data = distance[mask]

    if climate_model == "OAC":
        species_inventory = {
            'CO2': co2_emissions * 1e9,    # Mt → kg
            'Contrails': distance_data,
            'H2O': h2o_emissions * 1e9,
        }
    else:
         species_inventory = {
            'CO2': co2_emissions * 1e9,    # Mt → kg
            'Contrails': distance_data,
            'H2O': h2o_emissions * 1e9,
            'NOx - ST O3 increase': nox_emissions * 1e9,
            'NOx - CH4 decrease and induced': nox_emissions * 1e9,
            'Soot': soot_emissions * 1e9,
            'Sulfur': sulfur_emissions * 1e9,
        }   

    return species_inventory

#----Regionialisation functions----
def apply_region_weights_cont(lat_vals, lon_vals, dist_vals, region_weights):
    """
    Apply regional weights to defined regions to allow for rescaling.

    Parameters
    ----------
    lat_vals : array
        Latitude values from base inventory.
    lon_vals : array
        Longitude values from base inventory.
    dist_vals : array
        Distance travelled values extracted from base inventory.
    region_weights : dict
        Dictionary with assigned region weight.

    Returns
    -------
    weighted_dist : array
        array of the distance travelled after rescaling with weights.        


    """
    region_bounds = {
        "North America": [-180, -45, 12, 70],
        "Atlantic region": [-45, -25, 12, 70],
        "South America": [-90, -25, -60, 12],
        "Pacific": [-180, -90, -60, 12],
        "Far North": [-180, 60, 70, 90],
        "Europe": [-25, 60, 35, 70],
        "Africa and Middle East": [-25, 60, -35, 35],
        "Asia": [60, 180, -10, 90],
        "Oceania": [110, 180, -50, -10]
    }

    lon_converted = np.where(lon_vals > 180, lon_vals - 360, lon_vals)
    weighted_dist = dist_vals.copy().astype(np.float32)
    
    for region, bounds in region_bounds.items():
        if region not in region_weights:
            continue
        lon_min, lon_max, lat_min, lat_max = bounds
        weight = region_weights[region]
        region_mask = (
            (lon_converted >= lon_min) & (lon_converted <= lon_max) &
            (lat_vals >= lat_min) & (lat_vals <= lat_max)
        )
        weighted_dist[region_mask] *= weight

    return weighted_dist

def scaled_emissions_to_nc_with_weights_cont(input_nc, output_nc, aggdist, year, region_weights, chunk_size=50000):
    """
    Function to get rescaled inventory with desired regional weoghts applied.

    Parameters
    ----------
    input_nc : str
        File path to refernce inventory the rescaling will be based on.
    output_nc : str
        File path to output inventory.
    aggdist : float
        Numerical value of aggregated global distance flown.
    year : int
        Year the inventory represents.
    region_weights : dict
        Dictionary with the region weights.
    chunk_size : int
        size of discretisation of map.   

    Returns
    -------
    output_file : nc file
        Rescaled output inventory saved in folder "inputs"     
    """
    os.makedirs("oac_inputs", exist_ok=True)
    
    ds = xr.open_dataset(input_nc)
    total_points = len(ds['index'])
    
    original_total = float(ds['distance'].sum())
    scale_factor = aggdist / original_total
    weighted_cont = np.zeros(total_points, dtype=np.float32)
    
    for start_idx in range(0, total_points, chunk_size):
        end_idx = min(start_idx + chunk_size, total_points)
        lat_chunk = ds['lat'].isel(index=slice(start_idx, end_idx)).values
        lon_chunk = ds['lon'].isel(index=slice(start_idx, end_idx)).values
        dist_chunk = ds['distance'].isel(index=slice(start_idx, end_idx)).values * scale_factor
        weighted_chunk = apply_region_weights_cont(lat_chunk, lon_chunk, dist_chunk, region_weights)
        weighted_cont[start_idx:end_idx] = weighted_chunk
    
    current_total = np.sum(weighted_cont)
    final_cont = weighted_cont * (aggdist / current_total)
    
    new_ds = ds.copy()
    new_ds['distance'] = (('index',), final_cont)
    new_ds.attrs['Inventory_Year'] = year
    new_ds['distance'].attrs['long_name'] = 'distance flown'
    new_ds['distance'].attrs['units'] = 'km'
    new_ds.attrs['region_weights'] = str(region_weights)

    output_path = os.path.join("oac_inputs", output_nc)
    new_ds.to_netcdf(output_path)
    print(f"Saved weighted scaled emissions to {output_path}")
    ds.close()

# default reference inventory is the DLR 2025, can be changed to others
def weighted_cont(dist, start_year, end_year, region_weights,  step = 1, ref_inv = 'oac/repository/emi_inv_2025.nc'):
     """
     Function to generate family of rescaled inventories based on region weights.

     Parameters
     ----------
     dist : List
        List of total distance flown for each simulated year.
     start_yeat : int
        Starting year of dataset.
     end_year : int
        Last year of dataset.
     region_weights : dict
        Dictionary of region weights
     step : int
        Step between years. Default: 1.
     ref_inv : str
        File path to reference inventory.

     Returns
     -------
     All rescaled inventories dictated by inputs.  
        
     """
     years = [i for i in range(start_year,end_year+1,step)]
     
     if isinstance(region_weights, dict):
          final_region_weights = region_weights
          for i in range(len(years)):
            nc_name = f"dist_weighted_mat_generated_nc_{years[i]}.nc"
            scaled_emissions_to_nc_with_weights_cont(ref_inv, nc_name, dist[i], float(years[i]), final_region_weights)
     else:
          for idx, weights in enumerate(region_weights):
              for i in range(len(years)):
                nc_name = f"dist_weighted_mat_generated_nc_{years[i]}.nc"
                scaled_emissions_to_nc_with_weights_cont(ref_inv, nc_name, dist[i], float(years[i]), weights)


def generate_toml(start_year, end_year, step, output_file, specie_settings, 
                       inv_species=None, out_species=None, weighted=None, scaling= None, scale_file = None, inv_files=None,):
    """ Function to generate toml file to be able to run OAC

    Can be further edited to change species and model settings

    Parameters
    ---------
    start_year : float
        start year of simulation
    end_year : float
        end year of simulation
    step : float
        step years to take inbetween start and end year for simulation
    output_file : str
        file name for the toml file, need to inlcude .toml extension
    specie_settings : dict
        dictionary of efficacy values for that particular species    
    inv_species : list, optional
        input species list of inventories. Default list ["CO2","H2O", "NOx", "distance"]
    out_species : list, optional
        outpuyt species list. Default list ["CO2","H2O", "cont"]
    weighted : str, optional
        None- no weighted input inventories;
        "distance_weighted" - weighted distance values only to see effect of contrails.
    scaling : str, optional
        Use different time evolutions for oac;
        "scale" - scaling;
        "norm" - time norm evolution;
        None - no evolution, interpolation of inventories.
    scale_file : str, optional
        Depending on scale type, The appropriate nc scaling file.    
    inv_files : list, optional
        list of specific input inventories. otherwise it follows the deault names.   

    Returns
    -------
    toml file : .toml
        Returns a .toml file that instructs the oac run in a tomls folder


    """
    os.makedirs("tomls", exist_ok=True)
    
    # --- Raise fundamental errors ---
    if start_year >= end_year:
        raise ValueError("start_year must be < end_year")

    if step <= 0:
        raise ValueError("step must be positive")

    valid_weighted = {None, "weighted", "distance_weighted"}
    if weighted not in valid_weighted:
        raise ValueError(f"weighted must be one of {valid_weighted}")



    # --- Species Settings ---
    # Default settings:
    co2_lambda = 0.73 
    h2o_efficacy = 1.14
    O3_efficacy = 1.37
    PMO_efficacy =1.37
    CH4_efficacy = 1.14
    cont_efficacy = 0.59
    # rf method for co2 kept as default value for other species, need to check if it affects them. If not, no issue.
    # But if it does affect, co2 rf method will be placed in the model settings.
    co2_rf_method = "Etminan_2016"
    

    h = end_year - start_year + 1

    if inv_species is None:
        inv_species = ["CO2","H2O", "NOx", "distance"]
    else:
        inv_species = inv_species    
    if out_species is None:
        out_species = ["CO2","H2O", "cont"]
    else:
        out_species = out_species

    if inv_files is None:    
        if scaling == None:    
            if weighted is None:
                inventory_files = [f"mat_generated_nc_{year}.nc" for year in range(start_year, end_year, step)]
            elif weighted == "distance_weighted":
                inventory_files = [f"dist_weighted_mat_generated_nc_{year}.nc" for year in range(start_year, end_year, step)]
            elif weighted  == "weighted":
                inventory_files = [f"weighted_mat_generated_nc_{year}.nc" for year in range(start_year, end_year, step)]   
            time_evo = ""    

        elif scaling == "norm":
            if weighted is None:
                inventory_files = [f"mat_generated_nc_{start_year}.nc"]
            elif weighted == "distance_weighted":
                inventory_files = [f"dist_weighted_mat_generated_nc_{start_year}.nc"]
            elif weighted  == "weighted":
                inventory_files = [f"weighted_mat_generated_nc_{start_year}.nc" ]    

            time_evo = f'file = "{scale_file}"'
        
        elif scaling == "scale":
            if weighted is None:
                inventory_files = [f"mat_generated_nc_{year}.nc" for year in range(start_year, end_year, step)]
            elif weighted == "distance_weighted":
                inventory_files = [f"dist_weighted_mat_generated_nc_{year}.nc" for year in range(start_year, end_year, step)]
            elif weighted  == "weighted":
                inventory_files = [f"weighted_mat_generated_nc_{year}.nc" for year in range(start_year, end_year, step)]  

            time_evo = f'file = "{scale_file}"'
    elif inv_files is not None:
        inventory_files = inv_files
        time_evo = ""        

    if "CO2" in inv_species:
        co2_lambda = specie_settings.get("lambda",co2_lambda)
        co2_rf_method = specie_settings.get("rf_method","Etminan_2016")
    if "H2O" in inv_species:
        h2o_efficacy = specie_settings.get("efficacy",h2o_efficacy)
    if "distance" in inv_species:     
        cont_efficacy = specie_settings.get("efficacy",cont_efficacy)

    toml_content = f'''

    # Species considered
    [species]
    # Species defined in emission inventories
    # possible values: "CO2", "H2O", "NOx", "distance"
    inv = {inv_species}
    # Assumed NOx species in emission inventory
    # possible values: "NO", "NO2"
    nox = "NO"
    # Output / response species
    # possible values: "CO2", "H2O", "O3", "CH4", "PMO", "cont"
    out = {out_species}

    # Emission inventories                                                    
    [inventories]
    dir = "oac_inputs/"
    files = [ {", ".join([f'"{file}"' for file in inventory_files])}
    ]
    # base emission inventories, only considered if rel_to_base = true
    rel_to_base = false
    base.dir = "input/"
    base.files = [
        "rnd_inv_2020.nc",
        "rnd_inv_2030.nc",
        "rnd_inv_2040.nc",
        "rnd_inv_2050.nc",
    ]

    # Output options
    [output]
    # Full simulation run = true, climate metrics only = false
    full_run = true
    dir = "oac_results/"
    name = "gen_{start_year}s"
    overwrite = true
    # Computation of 2D concentration responses is not yet supported.
    # possible values: false 
    concentrations = false

    # Time settings
    [time]
    dir = "time_evo/"
    # Time range in years: t_start, t_end, step, (t_end not included)
    range = [{start_year}, {end_year}, {step}]
    # Time evolution of emissions
    # either type "scaling" or type "norm"
    {time_evo}
    #file = "time_scaling_example.nc"
    #file = "time_norm_example.nc"

    # Global background concentrations
    [background]
    dir = "oac/repository/"
    CO2.file = "co2_bg.nc"
    CO2.scenario = "SSP2-4.5"
    #CO2.scenario = "SSP1-1.9"
    #CO2.scenario = "SSP4-6.0"
    #CO2.scenario = "SSP3-7.0"
    CH4.file = "ch4_bg.nc"
    CH4.scenario = "SSP2-4.5"
    N2O.file = "n2o_bg.nc"
    N2O.scenario = "SSP2-4.5"

    # Response options
    [responses]
    dir = "oac/repository/"
    CO2.response_grid = "0D"
    # "Sausen&Schumann"
    CO2.conc.method = "Sausen&Schumann"
    # RF method based on Etminan et al. 2016 is used by default.
    # DEFAULT Etminan_2016
    CO2.rf.method = "{co2_rf_method}"

    H2O.response_grid = "2D"
    H2O.rf.file = "resp_RF.nc"    # AirClim response surface

    O3.response_grid = "2D"
    O3.rf.file = "resp_RF_O3.nc"  # tagging
    #O3.rf.file = "resp_RF.nc"    # AirClim response surface, requires adjustment of CORR_RF_O3 !

    CH4.response_grid = "2D"
    CH4.tau.file = "resp_ch4.nc"  # tagging
    CH4.rf.method = "Etminan_2016"

    cont.response_grid = "cont"
    cont.resp.file = "resp_cont_lf.nc"

    # Temperature options
    [temperature]
    # valid methods: "Boucher&Reddy"
    method = "Boucher&Reddy"
    # Climate sensitivity parameter, Ponater et al. 2006, Table 1
    # https://doi.org/10.1016/j.atmosenv.2006.06.036
    CO2.lambda = {co2_lambda}
    # Efficacies, Ponater et al. 2006, Table 1
    H2O.efficacy = {h2o_efficacy}
    O3.efficacy = {O3_efficacy}
    PMO.efficacy = {PMO_efficacy}
    CH4.efficacy = {CH4_efficacy}
    #default value 0.59 for contrail
    cont.efficacy = {cont_efficacy}

    # Climate metrics options
    [metrics]
    # iterate over elements in lists types t_0 and H
    types = ["AGWP", "ATR", "AGTP"] # valid climate metrics: AGTP, AGWP, ATR
    H = [{h}]                        # Time horizon, t_final = t_0 + H - 1
    t_0 = [{start_year}]                    # Start time for metrics calculation

    # aircraft defined in inventory
    # following identifiers are NOT allowed: "TOTAL"
    # "DEFAULT" is used if "ac" coordinate not defined in emission inventories
    # G_250, eff_fac and PMrel must be defined for each aircraft if contrails are
    # to be calculated.
    [aircraft]
    types = ["DEFAULT"]
    DEFAULT.G_250 = 1.70   # Schmidt-Appleman mixing line slope at 250 hPa
    DEFAULT.eff_fac = 1.0  # efficiency factor compared to 0.333
    DEFAULT.PMrel = 1.0    # relative PM emissions compared to 1e15
    '''


    
    
    file_path = os.path.join("tomls", output_file)    
    with open(file_path, 'w') as f:
        f.write(toml_content)
    print(f"TOML file written to {output_file}")