Installation guide for developers
--------------------------------
If you want to contribute to the development of AeroCM, you can clone the repository and install the package in a virtual environment using [uv](https://docs.astral.sh/uv/):

``` {.bash}
git clone https://github.com/AeroMAPS/AeroCM.git
cd aerocm
uv sync
```

If you also want to run the Jupyter notebooks developed for the reference paper, install the extra dependencies with this command:

``` {.bash}
uv sync --extra publications
```

Template to implement new climate models
--------------------------------------
You can find a template to implement new climate models in the [API reference](../api/aerocm.climate_models.template.md). It includes a basic structure of a climate model. You can copy this template and modify it according to your needs.
For more information, don't hesitate to reach out to us at [aeromaps@isae-supaero.fr](mailto:aeromaps@isae-supaero.fr).