import numpy as np


def EmissionProfile(start_year, t0, time_horizon, profile, unit_value):
    emissions = np.zeros(t0 - start_year + time_horizon + 1)
    if profile == "pulse":
        emissions[t0 - start_year] = unit_value
    elif profile == "step":
        for k in range(0, time_horizon + 1):
            emissions[t0 - start_year + k] = unit_value
    return emissions
