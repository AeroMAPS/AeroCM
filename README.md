# AeroCM

AeroCM is a package storing several climate models dedicated to aviation. It currently includes analytical (IPCC), metric-based (GWP*, LWE) and reduced-complexity (FaIR) models. The models have been standardised (species, species settings...) for allowing a generic use and comparisons. The models can be used in order to either directly assess the climate impacts induced by emission scenarios, or calculate aviation climate metrics (e.g. GWP, GTP, ATR).

AeroCM is licensed under the [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) license.

A [documentation](https://aeromaps.github.io/AeroCM/) is available for more details on AeroCM, including tutorials and API references.


Quick start
------------------

The use of the Python Package Index ([PyPI](https://pypi.org/project/aerocm/)) is the simplest method for installing AeroCM.

**Prerequisite**: AeroMAPS needs at least Python 3.10.

You can install the latest version with this command:

``` {.bash}
pip install --upgrade aerocm
```

If you also want to run the Jupyter notebooks developed for the reference paper, use the following command:

``` {.bash}
pip install --upgrade aerocm[publications]
```


For developers
------------------

If you want to contribute to the development of AeroCM, you can clone the repository and install the package in a virtual environment using [Poetry](https://python-poetry.org/):

``` {.bash}
git clone https://github.com/AeroMAPS/AeroCM.git
cd aerocm
poetry install
```

If you also want to run the Jupyter notebooks developed for the reference paper, install the extra dependencies with this command:

``` {.bash}
poetry install -E publications
```


Citation
--------

If you use AeroCM in your work, please cite the following reference.

> Planès, T., Pollet, F., Perini, M. (2025).
Aviation climate metrics: calculations and applications based on an open-source framework for standardised climate models.
Preprint.