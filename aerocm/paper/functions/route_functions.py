import numpy as np
from geographiclib.geodesic import Geodesic
from scipy.interpolate import RegularGridInterpolator

def great_circle_path(lat1, lon1, lat2, lon2, npoints=100, waypoint=None):
    """
    Calculates a great circle route, with the option of a waypoint.
    
    Args:
        lat1, lon1: coordinates of the starting point (°)
        lat2, lon2: coordinates of the destination point (°)
        npoints: number of points per segment
        waypoint: tuple (lat, lon) to force a waypoint (optional)
    
    Returns:
        lats (np.array): latitudes of the trajectory
        lons (np.array): longitudes of the trajectory
        dists (np.array): cumulative distances (km) along the trajectory
        total_distance_km (float): total distance of the trajectory (km)
    """
    geod = Geodesic.WGS84
    
    def segment(latA, lonA, latB, lonB, npoints):
        g = geod.Inverse(latA, lonA, latB, lonB)
        line = geod.Line(latA, lonA, g['azi1'])
        total_dist = g['s12']  # m
        dists = np.linspace(0, total_dist, npoints)
        
        lats, lons = [], []
        for d in dists:
            pos = line.Position(d)
            lats.append(pos['lat2'])
            lons.append(pos['lon2'])
        return np.array(lats), np.array(lons), dists / 1000.0, total_dist / 1000.0
    
    # Without waypoint
    if waypoint is None:
        lats, lons, dists, total_dist = segment(lat1, lon1, lat2, lon2, npoints)
        return lats, lons, total_dist
    
    # With waypoint (two segments=
    latw, lonw = waypoint
    lats1, lons1, dists1, dist1 = segment(lat1, lon1, latw, lonw, npoints)
    lats2, lons2, dists2, dist2 = segment(latw, lonw, lat2, lon2, npoints)
    
    # Fusion
    lats = np.concatenate([lats1, lats2[1:]])
    lons = np.concatenate([lons1, lons2[1:]])
    dists = np.concatenate([dists1, dist1 + dists2[1:]])
    total_distance_km = dist1 + dist2
    
    return lats, lons, total_distance_km
    
    
def mean_along_path(dataarray, lats, lons):
    """
    Calculates the average value of a field (lat, lon) along a trajectory,
    automatically handling coordinate conventions.
    
    Args:
        dataarray: xarray.DataArray 2D (lat, lon)
        lats, lons: arrays of route coordinates (great_circle_path outputs)
        
    Returns:
        float: average of values interpolated along the trajectory
        np.array: values interpolated along the trajectory
    """
    # Extraction of lat/lon
    lat_vals = dataarray['lat'].values
    lon_vals = dataarray['lon'].values
    field = dataarray.values

    # Lat verifications
    if lat_vals[0] > lat_vals[-1]:
        lat_vals = lat_vals[::-1]
        field = field[::-1, :]
    
    # Lon verifications
    if np.all(lon_vals >= 0):  
        lons_mod = np.mod(lons, 360)
    else:
        lons_mod = np.where(lons > 180, lons - 360, lons)
    
    # Interpolation
    interp = RegularGridInterpolator(
        (lat_vals, lon_vals), field,
        bounds_error=False, fill_value=np.nan
    )
    
    points = np.array([lats, lons_mod]).T

    values = interp(points)
    
    return np.nanmean(values), values