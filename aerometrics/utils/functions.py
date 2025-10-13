import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def emission_profile_function(
        start_year: int,
        t0: int,
        time_horizon: int,
        profile: str,
        unit_value: float
):
    """
    Generate an emission profile based on the specified type (e.g. pulse) and time horizon.
    :param start_year: start of the simulation
    :param t0: year of the emission event
    :param time_horizon: time horizon of the simulation after the emission event
    :param profile: type of emission profile ("pulse", "step", "1% growth")
    :param unit_value: magnitude of the emission event
    :return: emission profile as a numpy array
    """
    if profile not in ["pulse", "step", "1% growth"]:
        raise ValueError(f"Invalid profile '{profile}'. Choose from 'pulse', 'step', or '1% growth'.")
    emissions = np.zeros(t0 - start_year + time_horizon + 1)
    if profile == "pulse":
        emissions[t0 - start_year] = unit_value
    elif profile == "step":
        for k in range(0, time_horizon + 1):
            emissions[t0 - start_year + k] = unit_value
    elif profile == "1% growth":
        for k in range(0, time_horizon + 1):
            emissions[t0 - start_year + k] = unit_value * (1 + 0.01) ** k
    return emissions


def plot_simulation_results(
    ds: xr.Dataset,
    data_var: str,
    species: list[str] | None = None,
    title: str | None = None,
    figsize=(8, 5),
    stacked: bool = False,
    ax: plt.Axes | None = None,
    label_prefix: str | None = None,
):
    """
    Plot selected species over time for a chosen data variable.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with dimensions (species, year).
    data_var : str
        Name of the data variable to plot (e.g. 'temperature_change').
    species : list[str], optional
        Species names to plot. If None, all species are plotted.
    title : str, optional
        Custom plot title (ignored if ax is provided and already has one).
    figsize : tuple, optional
        Figure size in inches (only used if ax is None).
    stacked : bool, optional
        If True, plots a stacked area chart instead of individual lines.
    ax : matplotlib.axes.Axes, optional
        Existing matplotlib Axes to draw on. If None, a new figure is created.
    label_prefix : str, optional
        Optional prefix to add to species labels (useful when plotting multiple datasets).
    """

    # --- Validation ---
    if data_var not in ds.data_vars:
        raise ValueError(f"'{data_var}' not found in dataset. Available: {list(ds.data_vars)}")

    if species is not None:
        missing = [s for s in species if s not in ds.species.values]
        if missing:
            raise ValueError(f"Species not found in dataset: {missing}")
        da = ds[data_var].sel(species=species)
    else:
        da = ds[data_var]

    years = da.year.values
    data = da.transpose("year", "species").values  # shape (year, species)
    species_list = da.species.values

    # --- Prepare axes ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        new_fig = True
    else:
        new_fig = False

    # --- Plot ---
    if stacked:
        ax.stackplot(years, data.T, labels=[f"{label_prefix or ''}{sp}" for sp in species_list])
    else:
        for i, sp in enumerate(species_list):
            label = f"{label_prefix or ''}{sp}"
            ax.plot(years, data[:, i], label=label)

    # --- Format ---
    if new_fig:
        ax.set_xlabel("Year")
        ax.set_ylabel(data_var.replace("_", " ").title())
        ax.set_title(title or f"{data_var.replace('_', ' ').title()}")
        ax.legend(title="Species", loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
    else:
        # If reusing an existing axis, only update legend
        ax.legend(loc="upper left")

    return ax