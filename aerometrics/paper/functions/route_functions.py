import numpy as np
from geographiclib.geodesic import Geodesic
from scipy.interpolate import RegularGridInterpolator

def great_circle_path(lat1, lon1, lat2, lon2, npoints=100, waypoint=None):
    """
    Calcule une route orthodromique (great circle), avec possibilité d’un point de passage.
    
    Args:
        lat1, lon1 : coordonnées du point de départ (°)
        lat2, lon2 : coordonnées du point d’arrivée (°)
        npoints : nombre de points par segment
        waypoint : tuple (lat, lon) pour forcer un point de passage (optionnel)
    
    Returns:
        lats (np.array) : latitudes de la trajectoire
        lons (np.array) : longitudes de la trajectoire
        dists (np.array) : distances cumulées (km) le long de la trajectoire
        total_distance_km (float) : distance totale de la trajectoire (km)
    """
    geod = Geodesic.WGS84
    
    def segment(latA, lonA, latB, lonB, npoints):
        g = geod.Inverse(latA, lonA, latB, lonB)
        line = geod.Line(latA, lonA, g['azi1'])
        total_dist = g['s12']  # mètres
        dists = np.linspace(0, total_dist, npoints)
        
        lats, lons = [], []
        for d in dists:
            pos = line.Position(d)
            lats.append(pos['lat2'])
            lons.append(pos['lon2'])
        return np.array(lats), np.array(lons), dists / 1000.0, total_dist / 1000.0
    
    # Cas sans waypoint
    if waypoint is None:
        lats, lons, dists, total_dist = segment(lat1, lon1, lat2, lon2, npoints)
        return lats, lons, total_dist
    
    # Cas avec waypoint : deux segments
    latw, lonw = waypoint
    lats1, lons1, dists1, dist1 = segment(lat1, lon1, latw, lonw, npoints)
    lats2, lons2, dists2, dist2 = segment(latw, lonw, lat2, lon2, npoints)
    
    # Fusionner en enlevant le doublon du waypoint
    lats = np.concatenate([lats1, lats2[1:]])
    lons = np.concatenate([lons1, lons2[1:]])
    dists = np.concatenate([dists1, dist1 + dists2[1:]])
    total_distance_km = dist1 + dist2
    
    return lats, lons, total_distance_km
    
    
def mean_along_path(dataarray, lats, lons):
    """
    Calcule la valeur moyenne d'un champ (lat, lon) le long d'une trajectoire,
    en gérant automatiquement les conventions de coordonnées.
    
    Args:
        dataarray : xarray.DataArray 2D (lat, lon)
        lats, lons : arrays des coordonnées de la route (sorties great_circle_path)
        
    Returns:
        float : moyenne des valeurs interpolées le long de la trajectoire
        np.array : valeurs interpolées le long de la trajectoire
    """
    # Extraire lat/lon
    lat_vals = dataarray['lat'].values
    lon_vals = dataarray['lon'].values
    field = dataarray.values

    # 🔹 Vérif lat : doit être croissant pour l’interpolateur
    if lat_vals[0] > lat_vals[-1]:
        lat_vals = lat_vals[::-1]
        field = field[::-1, :]  # inverser l’ordre des latitudes
    
    # 🔹 Vérif lon : uniformiser
    if np.all(lon_vals >= 0):  
        # cas 0–360 → convertir les longitudes de la route
        lons_mod = np.mod(lons, 360)
    else:
        # cas -180–180 → convertir dans ce système
        lons_mod = np.where(lons > 180, lons - 360, lons)
    
    # Créer interpolateur
    interp = RegularGridInterpolator(
        (lat_vals, lon_vals), field,
        bounds_error=False, fill_value=np.nan
    )
    
    # Construire les points
    points = np.array([lats, lons_mod]).T
    
    # Interpoler
    values = interp(points)
    
    return np.nanmean(values), values